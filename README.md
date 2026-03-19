# GreenBoost — 3-Tier GPU Memory Extension for Linux

**Author:** Ferran Duarri
**License:** GPL v2 (open-source)
**Status:** First public release — after several months of testing on real hardware

<div align="center">

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/nvidia_greenboost)

</div>

---

I have been quietly working on this for a while and figured it was time to share it with the
community. This is a Linux kernel module + CUDA userspace shim that transparently extends
GPU VRAM using system DDR4 RAM and NVMe storage, so you can run large language models that
exceed your GPU memory without modifying the inference software at all.

**Important:** GreenBoost does not replace or modify the official NVIDIA kernel drivers
(`nvidia.ko`, `nvidia-uvm.ko`). It loads as a completely independent kernel module
alongside them and works at the CUDA allocation layer, not the driver layer.

---

## The problem I was trying to solve

I have an RTX 5070 with 12 GB VRAM and I wanted to run `glm-4.7-flash:q8_0`, which is a
31.8 GB model. The standard options are:

- **Offload layers to CPU** — works, but drops token/s by 5–10× because CPU RAM has no
  CUDA coherence. You end up waiting.
- **Use a smaller quantization** — you lose quality. At q4_0 the model is noticeably worse
  on reasoning tasks.
- **Buy a bigger GPU** — not realistic for consumer hardware. A 48 GB card costs more than
  a complete workstation.

None of those felt right, so I built an alternative: route the overflow memory to DDR4 via
DMA-BUF, which gives the GPU direct access to system RAM over PCIe 4.0 without a CPU
copy involved.

My system: i9-14900KF, RTX 5070 12 GB (Blackwell, compute capability 12.0), 64 GB
DDR4-3600, Samsung 990 EVO Plus 4 TB NVMe, Ubuntu 26.04, kernel 6.19, NVIDIA driver 580.x.

---

## How it works

Two components cooperate:

**1. Kernel module (`greenboost.ko`)**

Allocates pinned DDR4 pages using the buddy allocator (2 MB compound pages for efficiency)
and exports them as DMA-BUF file descriptors. The GPU can then import these pages as
CUDA external memory via `cudaImportExternalMemory`. From CUDA's perspective, those pages
look like device-accessible memory — it doesn't know they live in system RAM. The PCIe 4.0
x16 link handles the actual data movement (~32 GB/s).

A sysfs interface (`/sys/class/greenboost/greenboost/pool_info`) lets you monitor usage
live. A watchdog kernel thread monitors RAM and NVMe pressure and signals userspace before
things get dangerous.

**2. CUDA shim (`libgreenboost_cuda.so`, injected via `LD_PRELOAD`)**

Intercepts `cudaMalloc`, `cudaMallocAsync`, `cuMemAllocAsync`, `cudaFree`, and `cuMemFree`.
Small allocations (< 256 MB) pass straight through to the real CUDA runtime. Large ones
(KV cache, model weights overflowing VRAM) are redirected to the kernel module and imported
back as CUDA device pointers.

There is one tricky part worth mentioning: Ollama resolves GPU symbols via `dlopen` +
`dlsym` internally, which bypasses LD_PRELOAD on those symbols. To handle this, the shim
also intercepts `dlsym` itself (using `dlvsym` with the GLIBC version tag to bootstrap a
real pointer without recursion) and returns hooked versions of `cuDeviceTotalMem_v2` and
`nvmlDeviceGetMemoryInfo`. Without this, Ollama sees only 12 GB and puts layers on the CPU.

---

## Memory tiers

| Tier | Device | Capacity | Bandwidth | Used for |
|------|--------|----------|-----------|---------|
| T1 | RTX 5070 VRAM | 12 GB | ~336 GB/s | Hot layers, active compute |
| T2 | DDR4 pool | 51 GB | ~32 GB/s via PCIe 4.0 | KV cache, cold weights |
| T3 | NVMe swap | 64 GB | ~1.8 GB/s | Safety overflow (rarely hit) |

For the 32 GB model: T1 holds the hot layers, T2 holds the rest. T3 exists as a safety
net but in practice the model + KV cache at 32K context fits comfortably in T1+T2.

---

## What this code is based on 

 NVIDIA's open-source kernel module sourcecode had been explored
(`nvidia/open-gpu-kernel-modules`) to understand how UVM page fault handling works and how
CUDA external memory imports are expected to behave. That code was invaluable for getting
the DMA-BUF import path right. I want to be clear that I didn't copy any of it — I studied
the architecture and wrote my own implementation, but the public NVIDIA driver source was
a critical reference.

Also check at how `llama.cpp` handles CPU offload, how ExLlamaV2/V3 structures its KV
cache, and how vLLM's paged attention allocates memory. Understanding those patterns helped
figure out where in the allocation lifecycle to intercept.

---

## Installation

```bash
git clone https://gitlab.com/IsolatedOctopi/nvidia_greenboost.git
cd nvidia_greenboost

# Auto-detects GPU VRAM, RAM size, CPU P/E-core topology, NVMe capacity
# and computes optimal parameters at install time:
sudo ./greenboost_setup.sh full-install

# Offline / air-gapped workstation — skip the GitLab version check:
sudo ./greenboost_setup.sh full-install --skip-update-check

# After reboot, verify everything is working:
sudo ./greenboost_setup.sh diagnose
```

The installer runs the following steps automatically:

| Step | What it does |
|------|-------------|
| Purge | Stops Ollama/llama-server, unloads any previous GreenBoost module, removes old install artifacts |
| Dependencies | Installs build tools, kernel headers, CUDA dev packages |
| Build + install | Compiles `greenboost.ko` + `libgreenboost_cuda.so`, installs to `/lib/modules` and `/usr/local/lib` |
| Load | Inserts the kernel module with auto-detected pool sizes (T1/T2/T3) |
| System configs | Writes Ollama systemd env vars, NVMe udev rules, CPU governor service, hugepages, sysctl |
| **tune-all** | **CPU governor → performance; NVMe scheduler → none, read-ahead → 4 MB; THP → always; vm.swappiness=10; GRUB params (rcu_nocbs, nohz_full, transparent_hugepage=always); AI/compute libraries (OpenBLAS AVX2, hwloc, libnuma, nvtop, microcode)** |
| ExLlamaV3 | Sets up `/opt/greenboost/venv` with ExLlamaV3 + kvpress + trl |
| Restart | Restarts any services that were stopped during purge (Ollama, llama-server) |

---

## Quick usage examples

```bash
# Run Ollama as normal — GreenBoost is transparent
ollama run glm-4.7-flash:q8_0

# Monitor memory tiers live:
watch -n1 'cat /sys/class/greenboost/greenboost/pool_info'
```

---

## What GreenBoost is not

- It is **not** a replacement for the NVIDIA driver. `nvidia.ko`, `nvidia-uvm.ko`, and all
  NVIDIA official modules continue to run exactly as normal. GreenBoost loads beside them.
- It is **not** a virtual GPU. It doesn't expose a new GPU device or change how compute
  works. It only affects how CUDA memory allocations are routed.
- It is **not** a hack around driver restrictions. The DMA-BUF + external memory import
  path it uses is a documented CUDA feature.
- It does **not** work without the NVIDIA driver installed.

---

## Using GreenBoost inside containers, VMs, and WSL2

The CUDA shim supports a **kernel-module-free overflow path** (Path B) for
environments where `greenboost.ko` cannot be loaded — Docker, LXC, KVM guests,
WSL2, and shared HPC clusters.  When `/dev/greenboost` is unavailable the shim
automatically falls back to `cuMemHostRegister(DEVICEMAP)`, which pins anonymous
system RAM pages directly through the CUDA driver with no custom kernel module
required.

```bash
# Docker example
docker run --gpus all \
  -e GREENBOOST_ACTIVE=1 \
  -e LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
  -v /usr/local/lib/libgreenboost_cuda.so:/usr/local/lib/libgreenboost_cuda.so \
  my-llm-image python run_model.py

# Force Path B only (skip kernel-module path entirely):
GREENBOOST_USE_DMA_BUF=0 greenboost run python your_script.py

# Skip Path B, go straight to UVM:
GREENBOOST_NO_HOSTREG=1 greenboost run python your_script.py
```

The three overflow paths and when each is used:

| Path | Method | Requires kernel module | Best for |
|------|--------|----------------------|----------|
| A | DMA-BUF + `greenboost.ko` | Yes | Bare-metal Linux (default) |
| B | `cuMemHostRegister` no-kmod | No | Docker / LXC / KVM / WSL2 |
| C | `cuMemAllocManaged` UVM | No | Last resort / universal fallback |

For a full explanation of path selection, bandwidth trade-offs, and per-environment
instructions see **[container_vm_mode.md](container_vm_mode.md)**.

---

## Known limitations

- Tested on kernel 6.19 (Ubuntu 26.04). Earlier kernels may lack some exported symbols
  the module relies on.
- T3 NVMe tier is slow for random access. If a workload thrashes T3, performance degrades
  significantly. This is a fundamental NVMe latency constraint, not a software issue.
- Blackwell (compute capability 12.0) is the primary tested GPU. Ada Lovelace and Ampere
  should work.

---

## Usage

```
sudo ./greenboost_setup.sh [--skip-update-check] <command>

GLOBAL FLAGS:
  --skip-update-check  Skip GitLab version check (offline / air-gapped workstations)

COMMANDS:
  install     Build and install module + CUDA shim system-wide
  uninstall   Unload, remove module + all config files
  build       Build only (no system install)
  load        Load module with default 3-tier parameters
  unload      Unload module (keeps installed files)
  tune        Tune system for LLM workloads (governor, NVMe, THP, sysctl)
  tune-grub   Fix GRUB boot params (THP=always, rcu_nocbs, nohz_full…)
  tune-sysctl Consolidate sysctl files + apply compute-optimized knobs
  tune-libs   Install missing AI/compute libraries (OpenBLAS, hwloc…)
  tune-all    Run tune + tune-grub + tune-sysctl + tune-libs in sequence
  install-sys-configs  Install Ollama env, NVMe udev, CPU governor, hugepages, sysctl
  install-deps         Install all Ubuntu OS packages (build + CUDA + AI libs)
  setup-swap [GB]      Create/activate NVMe swap (default: auto-sized, ~64 GB for target model)
  full-install   Complete install — deps, module, shim, configs, tune-all, ExLlamaV3
  status      Show module status and 3-tier pool info
  diagnose    Full health check — run this after reboot to verify everything works
  optimize-model [--model M] [--strategy tensorrt|lora|exllama|all]
               Optimize LLM for max speed: TRT-LLM, LoRA, ExLlamaV3
  help        Show this help

CUDA SHIM WRAPPER:
  greenboost run <app> [args...]   Run a CUDA app with GreenBoost overflow enabled
```

** Greenboost_setup.sh had been adapted to detect if installation happens under RedHat based system and execute Alan's script. 


---
For full per-tool implementation instructions on llama.cpp / ik_llama.cpp check
**[implementing_further_into_tools.md](implementing_further_into_tools.md)**.

---

## Model optimization tools

As a next phase of the project, the [`tools/`](tools/) directory bundles several
open-source libraries that can help tune models further — quantization, KV cache
compression, fine-tuning, and alternative inference engines. See
**[tools/README.md](tools/README.md)** for details.

---

## License

GPL v2 — open-source. Attribution to Ferran Duarri is required in all forks, derivatives,
and any documentation that references this work.

```
Copyright (C) 2024-2026 Ferran Duarri
```

---

If you try this and run into issues, open an issue on the repository. I am particularly
interested in feedback from people running different GPU generations (Ada Lovelace, Ampere)
or different inference engines beyond Ollama and ExLlamaV3.

---

## Contributors

- **Alan Sill** ([@alansill](https://gitlab.com/alansill)) — contributed `greenboost_setup_rocky.sh`,
  a setup script for Red Hat-based systems (Rocky Linux, AlmaLinux, RHEL).

- **Jerry Nguyen** ([@jerry.nguyen](https://gitlab.com/jerry.nguyen)) — contributed the
  kernel-module-free overflow path (MR !3): `cuMemHostRegister(DEVICEMAP)` approach that
  enables GreenBoost VRAM extension inside containers, VMs, WSL2, and HPC clusters without
  requiring `greenboost.ko`.  Integrated as Path B of the blended shim in v2.5.


## Non direct Contributors

**v2.4** — UVM path improvements inspired by [HAMi-core](https://github.com/Project-HAMi/HAMi-core),
a Kubernetes GPU virtualization project that virtualizes VRAM for containers.
GreenBoost borrowed the `cuMemAdvise` prefetch pattern for its UVM fallback.