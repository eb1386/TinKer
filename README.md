# TinKer

Inference optimization for small transformer models on consumer NVIDIA GPUs.

## Benchmarks

268M param model, batch 1, seq 128, RTX 5080:

| Config | Latency | Speedup | Memory |
|---|---|---|---|
| PyTorch fp32 | 6.21 ms | 1.00x | 1105 MB |
| TinKer fp16 | 5.94 ms | 1.05x | 555 MB |
| TinKer fp16 + CUDA graph | 2.11 ms | **2.94x** | 570 MB |

## What it does

Makes models under 1B params run faster on RTX GPUs. One function call, no code changes needed.

On Linux with Triton: replaces attention, FFN, and norm modules with hand-written fused kernels. Fuses RoPE into the attention pass, SwiGLU into a single kernel, RMSNorm into one read-write. Then wraps everything in `torch.compile`.

On Windows (no Triton): skips module replacement (adds overhead without fusion). Instead applies fp16/bf16 casting, TF32 matmul, cuDNN benchmark, flash SDPA, and CUDA graph capture. The CUDA graph is the big one — records the whole forward pass and replays it, cutting out Python overhead entirely.

## Setup

Python 3.10+, PyTorch 2.0+, NVIDIA GPU.

```bash
git clone https://github.com/eb1386/TinKer.git
cd TinKer
pip install -e .
```

Or just copy the `tinker/` folder into your project.

Triton kernels (Linux only):
```bash
pip install -e ".[triton]"
```

## Usage

```python
from tinker import optimize

model = optimize(model, dtype=torch.float16, verbose=True)
```

With CUDA graph (static shapes, biggest speedup):

```python
model = optimize(
    model,
    dtype=torch.float16,
    cuda_graph=True,
    graph_input_shape=(1, 128),
    graph_vocab_size=32000,
    verbose=True,
)
```

All options:

```python
model = optimize(
    model,
    dtype=torch.float16,           # fp16 or bf16
    compile=True,                   # torch.compile (needs Triton)
    compile_mode="reduce-overhead",
    quantize=False,                 # int8 dynamic quantization
    replace_attention=True,         # patch attention (Triton only)
    replace_ffn=True,               # patch FFN (Triton only)
    replace_norm=True,              # patch norm (Triton only)
    rope_theta=10000.0,
    max_seq_len=2048,
    cuda_graph=True,
    graph_input_shape=(1, 128),
    graph_vocab_size=32000,
    verbose=True,
)
```

## Supported GPUs

NVIDIA only. No AMD, no integrated graphics. Auto-detects your card and loads a tuned config.

Has configs for every RTX 30/40/50 series card (3050 through 5090, all Ti/Super variants). Benchmarked on RTX 5080.

## Supported models

Anything using GQA/MHA/MQA + RoPE + SwiGLU + RMSNorm. Detects modules by weight structure, not class names.

Tested on [1386.ai](https://github.com/eb1386/1386.ai) Plasma 1.0 and 1.1. Compatible with LLaMA, Mistral, Qwen, Gemma.

Expects standard naming: `q_proj`/`k_proj`/`v_proj` or `wq`/`wk`/`wv`, `gate_proj`/`up_proj`/`down_proj` or `w1`/`w3`/`w2`.

## Limitations

- Inference only
- NVIDIA GPUs only
- Triton kernels need Linux. Windows gets CUDA tuning + graphs instead
- No KV cache in fused attention yet
- CUDA graphs need static input shapes
