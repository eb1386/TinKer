"""e2e benchmark"""

import copy
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinker import patch_model


# transformer components

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(x.dtype) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GQAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # rope tables
        half = head_dim // 2
        freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_table", torch.cos(angles), persistent=False)
        self.register_buffer("sin_table", torch.sin(angles), persistent=False)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """apply rope"""
        S = x.shape[2]
        cos = self.cos_table[:S].unsqueeze(0).unsqueeze(0)
        sin = self.sin_table[:S].unsqueeze(0).unsqueeze(0)
        out = torch.empty_like(x)
        out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, self.hidden_size)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = GQAAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# timing helper

NUM_WARMUP = 10
NUM_TIMED = 100


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

    # plasma config
    num_layers = 18
    hidden_size = 1024
    num_heads = 16
    num_kv_heads = 4
    head_dim = 64
    intermediate_size = 2816
    batch_size = 1
    seq_len = 128
    dtype = torch.bfloat16

    print("Model config: Plasma 1.0")
    print(f"  Layers: {num_layers}, Hidden: {hidden_size}, Heads: {num_heads}/{num_kv_heads}")
    print(f"  Head dim: {head_dim}, Intermediate: {intermediate_size}")
    print(f"  Batch: {batch_size}, Seq len: {seq_len}, Dtype: {dtype}")
    print()

    # baseline model
    baseline = SimpleTransformer(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    ).to(device=device, dtype=dtype).eval()

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    # baseline benchmark
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        baseline_ms = bench_fn(lambda: baseline(x), NUM_WARMUP, NUM_TIMED)
    baseline_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # patched model
    patched = copy.deepcopy(baseline)
    patch_model(patched, verbose=True)
    print()

    # patched benchmark
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        patched_ms = bench_fn(lambda: patched(x), NUM_WARMUP, NUM_TIMED)
    patched_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # results
    speedup = baseline_ms / patched_ms if patched_ms > 0 else float("inf")
    mem_saved = baseline_mem - patched_mem

    total_tokens = batch_size * seq_len

    print("=" * 60)
    print(f"{'':>30} {'Baseline':>14} {'TinKer':>14}")
    print("-" * 60)
    print(f"{'Median latency (ms)':>30} {baseline_ms:>14.3f} {patched_ms:>14.3f}")
    print(f"{'Throughput (tokens/s)':>30} {total_tokens / (baseline_ms / 1e3):>14.0f} {total_tokens / (patched_ms / 1e3):>14.0f}")
    print(f"{'Peak memory (MB)':>30} {baseline_mem:>14.1f} {patched_mem:>14.1f}")
    print("-" * 60)
    print(f"{'Speedup':>30} {speedup:>14.2f}x")
    print(f"{'Memory saved (MB)':>30} {mem_saved:>14.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
