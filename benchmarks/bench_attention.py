"""attention benchmark"""

import math
import sys
import time

import torch
import torch.nn.functional as F

from tinker.kernels.fused_attention import fused_attention, precompute_rope_tables


# pytorch reference

def _apply_rope_pytorch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """apply rope"""
    half = x.shape[-1] // 2
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)

    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """pytorch attention"""
    B, S, _ = q.shape
    num_kv_groups = num_heads // num_kv_heads

    q = q.view(B, S, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v = v.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3)

    # apply rope
    q = _apply_rope_pytorch(q, cos_table, sin_table)
    k = _apply_rope_pytorch(k, cos_table, sin_table)

    # gqa expansion
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

    # causal attention
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    return out.permute(0, 2, 1, 3).contiguous().view(B, S, num_heads * head_dim)


# benchmark configs

CONFIGS = [
    {"name": "Small",       "hidden": 512,  "num_heads": 8,  "num_kv_heads": 4},
    {"name": "Plasma 1.0",  "hidden": 1024, "num_heads": 16, "num_kv_heads": 4},
    {"name": "Plasma 1.1",  "hidden": 1280, "num_heads": 20, "num_kv_heads": 4},
    {"name": "LLaMA-like",  "hidden": 2048, "num_heads": 32, "num_kv_heads": 8},
]

BATCH_SIZES = [1, 4]
SEQ_LEN = 128
HEAD_DIM = 64
NUM_WARMUP = 10
NUM_TIMED = 100
DTYPE = torch.bfloat16


# timing helper

def bench_fn(fn, num_warmup: int, num_timed: int) -> float:
    """median latency"""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]


# main

def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(0)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu_name}")
    print()

    header = (
        f"{'Config':<25} {'PyTorch (ms)':>14} {'TinKer (ms)':>14}"
        f" {'Speedup':>10} {'Tokens/s':>14}"
    )
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]

        cos_table, sin_table = precompute_rope_tables(
            HEAD_DIM, SEQ_LEN, device=device, dtype=torch.float32,
        )

        for bs in BATCH_SIZES:
            label = f"{cfg['name']} bs={bs}"
            total_tokens = bs * SEQ_LEN

            q = torch.randn(bs, SEQ_LEN, num_heads * HEAD_DIM, device=device, dtype=DTYPE)
            k = torch.randn(bs, SEQ_LEN, num_kv_heads * HEAD_DIM, device=device, dtype=DTYPE)
            v = torch.randn(bs, SEQ_LEN, num_kv_heads * HEAD_DIM, device=device, dtype=DTYPE)

            pt_ms = bench_fn(
                lambda: pytorch_attention(
                    q, k, v, cos_table, sin_table, num_heads, num_kv_heads, HEAD_DIM
                ),
                NUM_WARMUP,
                NUM_TIMED,
            )
            tk_ms = bench_fn(
                lambda: fused_attention(
                    q.clone(), k.clone(), v.clone(),
                    cos_table, sin_table,
                    num_heads, num_kv_heads, HEAD_DIM,
                ),
                NUM_WARMUP,
                NUM_TIMED,
            )

            speedup = pt_ms / tk_ms if tk_ms > 0 else float("inf")
            throughput = total_tokens / (tk_ms / 1e3)

            print(
                f"{label:<25} {pt_ms:>14.4f} {tk_ms:>14.4f}"
                f" {speedup:>9.2f}x {throughput:>14.0f}"
            )

    print()
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak GPU memory allocated: {peak_mem:.1f} MB")


if __name__ == "__main__":
    main()
