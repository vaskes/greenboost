# Fork notice

**GreenBoost — created by Ferran Duarri**
<https://gitlab.com/IsolatedOctopi/nvidia_greenboost>

Upstream GitLab project is not reachable for automated contributions, so this
mirror exists to keep the code and carry the patches described below. Nothing
here changes the author's design; see `patches/` for the full delta.

## What this fork is

- **Upstream:** <https://gitlab.com/IsolatedOctopi/nvidia_greenboost>
  (GPL v2 open-source / dual-licensed; © 2026 Ferran Duarri)
- **Mirrored from:** <https://github.com/ClarkQAQ/nvidia_greenboost_mirror>
- **Mirror of the above by:** Kuduza Ai Lab — contact `vaskes`

`LICENSE` is upstream's, verbatim and unmodified. The dual-license terms and the
mandatory attribution apply to every file here.

## Why the patches exist

Two bugs, both found while running GreenBoost on an RTX 2080 Ti (sm_75) with
driver 615.71.09 / CUDA UMD 13.4 / kernel 7.0.0-38. Neither is specific to that
machine; both will bite anyone who `LD_PRELOAD`s the shim into a CUDA process.

### 1. `cuMemGetInfo_v2` returns `INVALID_CONTEXT` → ComfyUI never starts

`libcudart`'s `cudaMemGetInfo()` initialises the primary context and *then*
calls `cuMemGetInfo_v2()`. Symbol interposition routes that internal call
straight into `libcuda`, bypassing libcudart's initialisation:

```
File "/opt/ComfyUI/comfy/model_management.py", line 349, in get_total_memory
    _, mem_total_cuda = torch.cuda.mem_get_info(dev)
torch.AcceleratorError: CUDA error: invalid device context
Returning 201 (CUDA_ERROR_INVALID_CONTEXT) from cuMemGetInfo_v2
```

That is at import time, so the process dies before the server listens.

### 2. `cuMemHostGetDevicePointer` returns `INVALID_CONTEXT` → Path A dead

`cuMemHostRegister` returns `CUDA_SUCCESS`, `cuMemHostGetDevicePointer`
returns `201 INVALID_CONTEXT` — every size (64 MB – 20 GB), every flag
(`DEVICEMAP` / `0` / `PORTABLE`), with a valid context current, on bare metal
and in Docker alike. Path A and Path B both terminate in that call, so both
fail and the allocation falls back to UVM.

On UVA systems a `CU_MEMHOSTREGISTER_DEVICEMAP`-registered region is mapped
into the device address space at the same virtual address, so the host pointer
*is* the device pointer and the call is redundant. Confirmed with a standalone
binary on this driver: `devptr == hostptr` at every size and flag.

## Effect

20 GB allocation, RTX 2080 Ti, PCIe 3.0 x16:

| | throughput |
|---|---|
| before — Path A fails, falls back to UVM | 3.92 GB/s |
| after — Path A works | 11.06 – 11.40 GB/s |

Flat across 2 / 4 / 8 / 20 GB.

## Branch layout

| branch | contents |
|---|---|
| `main` | upstream `3ef86cd`, unmodified |
| `patches/ctx-and-uva` | the two fixes on top |

```bash
git checkout patches/ctx-and-uva
patch -p1 -i patches/0001-shim-uva-and-ctx.patch
```

59 lines, no behaviour change where the original calls already succeed — both
paths are strictly additive fallbacks.

## Unrelated observation

`real_cudaImportExternalMemory` / `real_cudaExternalMemoryGetMappedBuffer` /
`real_cudaDestroyExternalMemory` are resolved at lines 736-740 and never called;
the DMA-BUF fd is obtained and discarded. Importing it via
`cuImportExternalMemory` + `cuExternalMemoryGetMappedBuffer` returned
`999 CUDA_ERROR_UNKNOWN` on this driver, so it was left alone rather than
delete code that may be mid-way through being wired up.

## Testing

`probe/probe.c` (standalone, links `-lcuda`) measures Tier1 vs Tier2 bandwidth
by size; `probe/probe_shim.c` forces the shim's overflow path and confirms via
`/sys/class/greenboost/greenboost/pool_info` that the buffer really landed in
Tier 2. Both were used to produce the numbers above.
