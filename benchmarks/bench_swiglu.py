"""swiglu benchmark"""

import sys
import time

import torch
import torch.nn.functional as F

from tinker.kernels.fused_swiglu import fused_swiglu


# pytorch reference

def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """pytorch swiglu"""
    return F.silu(gate) * up


# benchmark configs

CONFIGS = [
    {"name": "Small",       "hidden": 512,  "intermediate": 1376},
    {"name": "Plasma 1.0",  "hidden": 1024, "intermediate": 2816},
    {"name": "Plasma 1.1",  "hidden": 1280, "intermediate": 3584},
    {"name": "LLaMA-like",  "hidden": 2048, "intermediate": 5504},
]

BATCH_SIZES = [1, 4]
SEQ_LEN = 128
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

    header = f"{'Config':<25} {'PyTorch (ms)':>14} {'TinKer (ms)':>14} {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        intermediate = cfg["intermediate"]
        for bs in BATCH_SIZES:
            label = f"{cfg['name']} bs={bs}"
            gate = torch.randn(bs, SEQ_LEN, intermediate, device=device, dtype=DTYPE)
            up = torch.randn(bs, SEQ_LEN, intermediate, device=device, dtype=DTYPE)

            pt_ms = bench_fn(lambda: pytorch_swiglu(gate, up), NUM_WARMUP, NUM_TIMED)
            tk_ms = bench_fn(lambda: fused_swiglu(gate, up), NUM_WARMUP, NUM_TIMED)
            speedup = pt_ms / tk_ms if tk_ms > 0 else float("inf")

            print(f"{label:<25} {pt_ms:>14.4f} {tk_ms:>14.4f} {speedup:>9.2f}x")

    print()
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak GPU memory allocated: {peak_mem:.1f} MB")


if __name__ == "__main__":
    main()
