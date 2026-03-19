# GreenBoost — integration guide for inference tools

## Static-CUDA vs lazy-CUDA apps

GreenBoost needs to be active before CUDA symbols are resolved. How that happens depends
on whether the app links libcuda at load time or resolves it later via `dlopen`:

| Category | Examples | How to activate |
|----------|----------|-----------------|
| **Static-CUDA** — links libcuda at startup | llama.cpp, llama-server, ik_llama.cpp, whisper.cpp | `LD_PRELOAD` alone is enough; shim activates at process start |
| **Lazy-CUDA** — dlopen libcuda at first use | Ollama, vLLM, PyTorch, TGI, CTranslate2, ExLlamaV3 | Must also set `GREENBOOST_ACTIVE=1`; use `greenboost run` or a systemd `Environment=` line |

For the dlsym hook (virtual VRAM reporting) to work, `GREENBOOST_ACTIVE=1` is also
required even for static-CUDA apps — llama.cpp and ik_llama.cpp use `dlopen`+`dlsym`
for their NVML backend, which bypasses LD_PRELOAD on those symbols.

**Rule of thumb:** always set `GREENBOOST_ACTIVE=1`. Use `greenboost run` for CLI
invocations, or add `Environment="GREENBOOST_ACTIVE=1"` to systemd service units.

---

## Setup

Ensure the shim is active for the process:

```bash
export LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so
export GREENBOOST_ACTIVE=1
```

Or use the installed wrapper (sets both automatically):

```bash
greenboost run llama-cli ...
greenboost run llama-server ...
greenboost run python your_script.py ...
```

---

## llama.cpp

```bash
greenboost run llama-cli \
    --model /opt/models/glm-4.7-flash-q8_0.gguf \
    --n-gpu-layers 999 \
    --ctx-size 131072 \
    --prompt "Hello"
```

Server mode:

```bash
greenboost run llama-server \
    --model /opt/models/glm-4.7-flash-q8_0.gguf \
    --n-gpu-layers 999 \
    --ctx-size 131072 \
    --port 8080
```

`--n-gpu-layers 999` is required — without it llama.cpp defaults to CPU-only.
With GreenBoost reporting 63 GB virtual VRAM, 999 schedules all layers to T1+T2.

---

## ik_llama.cpp

Same flags — ik_llama.cpp is a fork of llama.cpp and uses the same interface:

```bash
greenboost run ik_llama \
    --model /opt/models/glm-4.7-flash-q8_0.gguf \
    --n-gpu-layers 999 \
    --ctx-size 131072 \
    --prompt "Hello"
```

---

## vLLM

vLLM loads libcuda lazily through PyTorch (`torch.cuda` → `ctypes.CDLL`). Requires
`GREENBOOST_ACTIVE=1`.

```bash
greenboost run python -m vllm.entrypoints.openai.api_server \
    --model /opt/models/glm-4.7-flash-hf \
    --dtype float16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95
```

As a systemd service, add to the unit:

```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
Environment="GREENBOOST_ACTIVE=1"
```

---

## PyTorch scripts

Any `import torch; torch.cuda.*` call triggers lazy loading of libcuda via ctypes.

```bash
greenboost run python your_training_script.py
greenboost run python your_inference_script.py
```

Or inline without the wrapper:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python your_script.py
```

---

## text-generation-inference (TGI)

TGI uses a PyTorch backend — same lazy-CUDA pattern:

```bash
greenboost run text-generation-launcher \
    --model-id /opt/models/glm-4.7-flash-hf \
    --num-shard 1 \
    --max-total-tokens 131072
```

As a systemd service:

```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
Environment="GREENBOOST_ACTIVE=1"
```

---

## CTranslate2

CTranslate2 loads libcudart via ctypes in its Python bindings:

```bash
greenboost run python -c "
import ctranslate2
translator = ctranslate2.Translator('/opt/models/my-model', device='cuda')
results = translator.translate_batch([['Hello']])
print(results[0].hypotheses[0])
"
```

Or wrap the calling script:

```bash
greenboost run python your_ctranslate2_script.py
```

---

## ExLlamaV3

ExLlamaV3 is a Python framework — it loads CUDA through PyTorch:

```bash
greenboost run python your_exllama_script.py
```

When using the GreenBoost venv:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    /opt/greenboost/venv/bin/python your_exllama_script.py
```

The GreenBoost-patched KV cache layer (`CacheLayer_greenboost`) allocates directly
from `/dev/greenboost` and does not go through the shim's cudaMalloc path for cache
tensors — `GREENBOOST_ACTIVE=1` is still needed for virtual VRAM reporting.

---

## Ollama

Ollama is handled automatically by `install-sys-configs`, which injects the shim and
`GREENBOOST_ACTIVE=1` into its systemd unit. No manual wrapper needed.

```bash
ollama run glm-4.7-flash:q8_0   # GreenBoost is transparent
```

If running Ollama outside systemd:

```bash
greenboost run ollama serve
```

---

## Verify GreenBoost is active

```bash
# Should show virtual VRAM (63 GB), not physical (12 GB):
GREENBOOST_DEBUG=1 greenboost run llama-cli \
    --model /opt/models/glm-4.7-flash-q8_0.gguf \
    --n-gpu-layers 999 -p "hi" 2>&1 \
    | grep -i "greenboost\|vram\|gpu layers"

# Check kernel module pool after inference:
cat /sys/class/greenboost/greenboost/pool_info
```

---
