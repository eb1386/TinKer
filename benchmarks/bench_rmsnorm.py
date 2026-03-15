"""rmsnorm benchmark"""

import sys
import time

import torch

from tinker.kernels.fused_rmsnorm import fused_rmsnorm


# pytorch reference

def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """pytorch rmsnorm"""
    norm = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x * norm).to(x.dtype) * weight


# benchmark configs

CONFIGS = [
    {"name": "Small",       "dim": 512},
    {"name": "Plasma 1.0",  "dim": 1024},
    {"name": "Plasma 1.1",  "dim": 1280},
    {"name": "LLaMA-like",  "dim": 2048},
]

BATCH_SIZES = [1, 4]
SEQ_LEN = 128
NUM_WARMUP = 10
NUM_TIMED = 100
EPS = 1e-6
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

    header = f"{'Config':<25} {'PyTorch (ms)':>14} {'TinKer (ms)':>14} {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        dim = cfg["dim"]
        for bs in BATCH_SIZES:
            label = f"{cfg['name']} bs={bs}"
            x = torch.randn(bs, SEQ_LEN, dim, device=device, dtype=DTYPE)
            weight = torch.ones(dim, device=device, dtype=DTYPE)

            pt_ms = bench_fn(lambda: pytorch_rmsnorm(x, weight, EPS), NUM_WARMUP, NUM_TIMED)
            tk_ms = bench_fn(lambda: fused_rmsnorm(x, weight, EPS), NUM_WARMUP, NUM_TIMED)
            speedup = pt_ms / tk_ms if tk_ms > 0 else float("inf")

            print(f"{label:<25} {pt_ms:>14.4f} {tk_ms:>14.4f} {speedup:>9.2f}x")

    print()
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak GPU memory allocated: {peak_mem:.1f} MB")


if __name__ == "__main__":
    main()
