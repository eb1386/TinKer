"""model optimizer — the main tinker api

applies all available optimizations:
  1. module patching (swap to tinker fused modules — triton only)
  2. half precision (bf16/fp16)
  3. torch.compile with inductor (when triton available)
  4. cuda backend tuning (tf32, cudnn benchmark, sdpa)
  5. optional int8 quantization
  6. cuda graph capture (static shapes)
"""

import torch
import torch.nn as nn

from tinker.patch import patch_model

_HAS_TRITON = False
try:
    import triton
    _HAS_TRITON = True
except ImportError:
    pass

_HAS_COMPILE = hasattr(torch, "compile")


def optimize(
    model: nn.Module,
    dtype: torch.dtype | None = None,
    compile: bool | None = None,
    compile_mode: str = "reduce-overhead",
    quantize: bool = False,
    replace_attention: bool = True,
    replace_ffn: bool = True,
    replace_norm: bool = True,
    rope_theta: float = 10000.0,
    max_seq_len: int = 2048,
    cuda_graph: bool = False,
    graph_input_shape: tuple | None = None,
    graph_vocab_size: int = 32000,
    verbose: bool = False,
) -> nn.Module:
    """optimize a model for inference

    one call to speed up any compatible model. patches modules
    when triton is available, casts dtype, compiles, and tunes cuda.
    without triton, applies dtype/cuda optimizations that provide
    real speedups without wrapper overhead.
    """
    model.eval()

    if _HAS_TRITON:
        if verbose:
            print("[tinker] patching modules (triton fused kernels)...")
        model = patch_model(
            model,
            replace_attention=replace_attention,
            replace_ffn=replace_ffn,
            replace_norm=replace_norm,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            verbose=verbose,
        )
    elif verbose:
        print("[tinker] skipping module patching (no triton)")

    if dtype is not None:
        if verbose:
            print(f"  [tinker] casting to {dtype}")
        model = model.to(dtype=dtype)

    if quantize:
        if verbose:
            print("  [tinker] int8 quantization")
        model = _quantize_linears(model, verbose=verbose)

    should_compile = compile if compile is not None else _HAS_TRITON
    if should_compile and _HAS_COMPILE and _HAS_TRITON:
        if verbose:
            print(f"  [tinker] torch.compile (mode={compile_mode})")
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception as e:
            if verbose:
                print(f"  [tinker] compile failed: {e}")
    elif verbose and not _HAS_TRITON:
        print("  [tinker] skipping torch.compile (no triton)")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if verbose:
            print("  [tinker] cuda: tf32 + cudnn benchmark + flash sdpa")

    for p in model.parameters():
        p.requires_grad_(False)
    if verbose:
        print("  [tinker] disabled gradients")

    if cuda_graph and torch.cuda.is_available():
        if graph_input_shape is not None:
            model = _capture_cuda_graph(
                model, graph_input_shape, graph_vocab_size, verbose=verbose
            )
        elif verbose:
            print("  [tinker] skipping cuda graph (no input shape)")

    if verbose:
        info = _get_backend_info()
        print(f"\n  [tinker] ready ({info})")

    return model


def _quantize_linears(model: nn.Module, verbose: bool = False) -> nn.Module:
    """dynamic int8 quantization"""
    try:
        from torch.ao.quantization import quantize_dynamic
        model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        if verbose:
            count = sum(1 for m in model.modules()
                       if type(m).__name__ == "DynamicQuantizedLinear")
            print(f"  [tinker] quantized {count} linear layers")
    except Exception as e:
        if verbose:
            print(f"  [tinker] quantization failed: {e}")
    return model


class _CUDAGraphWrapper(nn.Module):
    """wraps model with cuda graph replay"""

    def __init__(self, model, static_input, static_output, graph):
        super().__init__()
        self._model = model
        self._static_input = static_input
        self._static_output = static_output
        self._graph = graph
        self._input_shape = static_input.shape

    def forward(self, x):
        if x.shape == self._input_shape:
            self._static_input.copy_(x)
            self._graph.replay()
            return self._static_output.clone()
        return self._model(x)


def _capture_cuda_graph(model, input_shape, vocab_size, verbose=False):
    """capture cuda graph"""
    try:
        device = next(model.parameters()).device
        static_input = torch.randint(0, vocab_size, input_shape, device=device)

        for _ in range(3):
            _ = model(static_input)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = model(static_input)

        if verbose:
            print(f"  [tinker] cuda graph captured (shape={input_shape})")
        return _CUDAGraphWrapper(model, static_input, static_output, graph)
    except Exception as e:
        if verbose:
            print(f"  [tinker] cuda graph failed: {e}")
        return model


def _get_backend_info() -> str:
    """describe active optimizations"""
    parts = []
    if _HAS_TRITON:
        parts.append("triton kernels")
    else:
        parts.append("cuda tuning + dtype")
    if torch.cuda.is_available():
        parts.append(torch.cuda.get_device_name(0))
    else:
        parts.append("cpu")
    return " + ".join(parts)


@torch.inference_mode()
def benchmark_model(
    model: nn.Module,
    input_shape: tuple = (1, 128),
    vocab_size: int = 32000,
    warmup: int = 10,
    runs: int = 100,
    device: str | None = None,
) -> dict:
    """benchmark forward pass latency"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = torch.randint(0, vocab_size, input_shape, device=device)

    for _ in range(warmup):
        _ = model(tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    import time
    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    tps = input_shape[0] * input_shape[1] / (median / 1000)

    result = {
        "median_ms": round(median, 3),
        "mean_ms": round(mean, 3),
        "min_ms": round(times[0], 3),
        "tokens_per_sec": round(tps),
        "device": device,
    }

    if torch.cuda.is_available():
        result["peak_memory_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 1
        )
        result["gpu"] = torch.cuda.get_device_name(0)

    return result
