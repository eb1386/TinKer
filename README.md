# TinKer

inference optimization for small transformer models on consumer nvidia gpus.

## benchmarks

268M param model, batch 1, seq 128, rtx 5080:

| config | latency | speedup | memory |
|---|---|---|---|
| pytorch fp32 | 6.21 ms | 1.00x | 1105 MB |
| tinker fp16 | 5.94 ms | 1.05x | 555 MB |
| tinker fp16 + cuda graph | 2.11 ms | **2.94x** | 570 MB |

## what it does

makes models under 1B params run faster on rtx gpus. one function call, no code changes needed.

on linux with triton: replaces attention, ffn, and norm modules with hand-written fused kernels. fuses rope into the attention pass, swiglu into a single kernel, rmsnorm into one read-write. then wraps everything in `torch.compile`.

on windows (no triton): skips module replacement (adds overhead without fusion). instead applies fp16/bf16 casting, tf32 matmul, cudnn benchmark, flash sdpa, and cuda graph capture. the cuda graph is the big one — records the whole forward pass and replays it, cutting out python overhead entirely.

## setup

python 3.10+, pytorch 2.0+, nvidia gpu.

```bash
git clone https://github.com/eb1386/TinKer.git
cd TinKer
pip install -e .
```

or just copy the `tinker/` folder into your project.

triton kernels (linux only):
```bash
pip install -e ".[triton]"
```

## usage

```python
from tinker import optimize

model = optimize(model, dtype=torch.float16, verbose=True)
```

with cuda graph (static shapes, biggest speedup):

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

all options:

```python
model = optimize(
    model,
    dtype=torch.float16,           # fp16 or bf16
    compile=True,                   # torch.compile (needs triton)
    compile_mode="reduce-overhead",
    quantize=False,                 # int8 dynamic quantization
    replace_attention=True,         # patch attention (triton only)
    replace_ffn=True,               # patch ffn (triton only)
    replace_norm=True,              # patch norm (triton only)
    rope_theta=10000.0,
    max_seq_len=2048,
    cuda_graph=True,
    graph_input_shape=(1, 128),
    graph_vocab_size=32000,
    verbose=True,
)
```

## supported gpus

nvidia only. no amd, no integrated graphics. auto-detects your card and loads a tuned config.

has configs for every rtx 30/40/50 series card (3050 through 5090, all ti/super variants). benchmarked on rtx 5080.

## supported models

anything using gqa/mha/mqa + rope + swiglu + rmsnorm. detects modules by weight structure, not class names.

tested on [1386.ai](https://github.com/eb1386/1386.ai) plasma 1.0 and 1.1. compatible with llama, mistral, qwen, gemma.

expects standard naming: `q_proj`/`k_proj`/`v_proj` or `wq`/`wk`/`wv`, `gate_proj`/`up_proj`/`down_proj` or `w1`/`w3`/`w2`.

## limitations

- inference only
- nvidia gpus only
- triton kernels need linux. windows gets cuda tuning + graphs instead
- no kv cache in fused attention yet
- cuda graphs need static input shapes
