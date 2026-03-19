# GreenBoost — llama.cpp / ik_llama.cpp integration

**Author:** Ferran Duarri

llama.cpp and ik_llama.cpp resolve GPU functions via `dlopen`+`dlsym` at runtime,
which bypasses `LD_PRELOAD` hooks on the symbols directly. GreenBoost intercepts
`dlsym` itself so the virtual VRAM size and allocation hooks stay active.

Additionally, both tools require `--n-gpu-layers` to schedule layers to GPU. Without
it they default to CPU-only. With GreenBoost reporting 63 GB virtual VRAM, passing
`--n-gpu-layers 999` schedules all layers to T1+T2 automatically.

---

## Setup

Ensure the shim is active for the process:

```bash
export LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so
export GREENBOOST_ACTIVE=1
```

Or run via the wrapper:

```bash
greenboost-run llama-cli ...
greenboost-run llama-server ...
greenboost-run ik_llama ...
```

---

## llama.cpp

```bash
LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
GREENBOOST_ACTIVE=1 \
llama-cli \
    --model /opt/models/glm-4.7-flash-q8_0.gguf \
    --n-gpu-layers 999 \
    --ctx-size 131072 \
    --prompt "Hello"
```

For the server:

```bash
LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
GREENBOOST_ACTIVE=1 \
llama-server \
    --model /opt/models/glm-4.7-flash-q8_0.gguf \
    --n-gpu-layers 999 \
    --ctx-size 131072 \
    --port 8080
```

---

## ik_llama.cpp

Same flags — ik_llama.cpp is a fork of llama.cpp and uses the same interface:

```bash
LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
GREENBOOST_ACTIVE=1 \
ik_llama \
    --model /opt/models/glm-4.7-flash-q8_0.gguf \
    --n-gpu-layers 999 \
    --ctx-size 131072 \
    --prompt "Hello"
```

---

## Verify GreenBoost is active

```bash
# Should show virtual VRAM (63 GB), not physical (12 GB):
GREENBOOST_DEBUG=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    llama-cli --model /opt/models/glm-4.7-flash-q8_0.gguf --n-gpu-layers 999 -p "hi" 2>&1 \
    | grep -i "greenboost\|vram\|gpu layers"

# Check kernel module pool after inference:
cat /sys/class/greenboost/greenboost/pool_info
```

---

## License

**GPL v2** — mandatory attribution to Ferran Duarri in all forks and derivatives.

```
Copyright (C) 2024-2026 Ferran Duarri
SPDX-License-Identifier: GPL-2.0-only
```
