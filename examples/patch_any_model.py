"""patch generic model"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# make importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from tinker import optimize, benchmark_model


# define model


class RMSNorm(nn.Module):
    """rmsnorm layer"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class Attention(nn.Module):
    """gqa attention"""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # expand kv
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn)


class SwiGLU(nn.Module):
    """swiglu ffn"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 intermediate_size: int):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = Attention(hidden_size, num_heads, num_kv_heads)
        self.ffn_norm = RMSNorm(hidden_size)
        self.feed_forward = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256,
                 num_layers: int = 4, num_heads: int = 8, num_kv_heads: int = 2,
                 intermediate_size: int = 512, max_seq_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_kv_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)


# build and patch

def main():
    print("Building toy transformer model...")
    model = ToyTransformer(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        intermediate_size=512,
        max_seq_len=512,
    )

    # optimize with tinker — one call does everything:
    # patches modules, compiles with torch.compile, sets dtype, enables tf32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # baseline benchmark
    print("\nbenchmarking baseline...")
    baseline = benchmark_model(model, input_shape=(1, 32), vocab_size=1000, warmup=5, runs=20)
    for k, v in baseline.items():
        print(f"  {k}: {v}")

    model = optimize(
        model,
        dtype=torch.float16 if device == "cuda" else None,
        rope_theta=10000.0,
        max_seq_len=512,
        verbose=True,
    )

    # verify shapes
    tokens = torch.randint(0, 1000, (1, 32), device=device)

    with torch.inference_mode():
        logits = model(tokens)

    print(f"\nInput shape:  {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (1, 32, 1000), f"unexpected shape: {logits.shape}"
    print("shape verification passed")

    # optimized benchmark
    print("\nbenchmarking optimized...")
    results = benchmark_model(model, input_shape=(1, 32), vocab_size=1000, warmup=5, runs=20)
    for k, v in results.items():
        print(f"  {k}: {v}")

    # compare
    if "median_ms" in baseline and "median_ms" in results:
        speedup = baseline["median_ms"] / results["median_ms"]
        print(f"\n  speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
