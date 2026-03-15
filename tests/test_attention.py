"""attention tests"""

import math

import pytest
import torch
import torch.nn.functional as F

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


# reference implementation

def precompute_rope_tables_ref(head_dim, max_seq_len, theta=10000.0, device=None):
    """rope table computation"""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0):
    """apply rope"""
    _B, _H, S, D = x.shape
    half = D // 2

    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()

    cos_vals = cos[start_pos:start_pos + S, :half].float()
    sin_vals = sin[start_pos:start_pos + S, :half].float()

    cos_vals = cos_vals.unsqueeze(0).unsqueeze(0)
    sin_vals = sin_vals.unsqueeze(0).unsqueeze(0)

    new_even = x_even * cos_vals - x_odd * sin_vals
    new_odd = x_even * sin_vals + x_odd * cos_vals

    # interleave back
    out = torch.stack([new_even, new_odd], dim=-1).flatten(-2)
    return out.to(x.dtype)


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    start_pos: int = 0,
) -> torch.Tensor:
    """pytorch reference"""
    B, S, _ = q.shape

    # reshape heads
    q = q.view(B, S, num_heads, head_dim).permute(0, 2, 1, 3)
    k = k.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v = v.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3)

    # apply rope
    q = apply_rope_ref(q, cos, sin, start_pos)
    k = apply_rope_ref(k, cos, sin, start_pos)

    # expand kv
    num_kv_groups = num_heads // num_kv_heads
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

    # causal attention
    out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)

    # reshape output
    out = out.to(q.dtype).permute(0, 2, 1, 3).contiguous().view(B, S, num_heads * head_dim)
    return out


# test configurations

ATTENTION_CONFIGS = [
    pytest.param(256, 4, 2, 64, 32, 1, id="tiny-gqa"),
    pytest.param(1024, 16, 4, 64, 64, 1, id="plasma1.0"),
    pytest.param(1280, 20, 4, 64, 64, 1, id="plasma1.1"),
    pytest.param(512, 8, 8, 64, 64, 1, id="mha-no-gqa"),
    pytest.param(512, 8, 1, 64, 64, 1, id="mqa-single-kv"),
]

EDGE_CONFIGS = [
    pytest.param(256, 4, 2, 64, 1, 1, id="edge-seq1"),
    pytest.param(256, 4, 2, 64, 128, 1, id="edge-seq128"),
]


# kernel tests

@skip_no_cuda
@pytest.mark.parametrize("hidden, num_heads, num_kv_heads, head_dim, seq_len, batch",
                         ATTENTION_CONFIGS + EDGE_CONFIGS)
def test_fused_attention_kernel(hidden, num_heads, num_kv_heads, head_dim, seq_len, batch):
    """kernel correctness"""
    from tinker.kernels.fused_attention import fused_attention, precompute_rope_tables

    torch.manual_seed(42)
    device = torch.device("cuda")

    q = torch.randn(batch, seq_len, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)

    cos, sin = precompute_rope_tables(head_dim, max_seq_len=max(seq_len, 256), device=device)

    expected = attention_ref(
        q.clone(), k.clone(), v.clone(),
        cos, sin,
        num_heads, num_kv_heads, head_dim,
    )
    actual = fused_attention(
        q.clone(), k.clone(), v.clone(),
        cos, sin,
        num_heads, num_kv_heads, head_dim,
    )

    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)


# module tests

@skip_no_cuda
@pytest.mark.parametrize("hidden, num_heads, num_kv_heads, head_dim, seq_len, batch",
                         ATTENTION_CONFIGS)
def test_tinker_attention_module(hidden, num_heads, num_kv_heads, head_dim, seq_len, batch):
    """module correctness"""
    from tinker.modules import TinKerAttention

    torch.manual_seed(42)
    device = torch.device("cuda")

    module = TinKerAttention(
        hidden_size=hidden,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max(seq_len, 256),
    ).to(device).to(torch.bfloat16)

    x = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.bfloat16)

    # reference path
    q = module.q_proj(x)
    k = module.k_proj(x)
    v = module.v_proj(x)

    attn_ref_out = attention_ref(
        q.clone(), k.clone(), v.clone(),
        module.cos_table, module.sin_table,
        num_heads, num_kv_heads, head_dim,
    )
    expected = module.o_proj(attn_ref_out)

    actual = module(x)

    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)


# from_module test

class _DummyAttention(torch.nn.Module):
    """dummy attention"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)


@skip_no_cuda
def test_from_module():
    """from_module correctness"""
    from tinker.modules import TinKerAttention

    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden, num_heads, num_kv_heads, head_dim = 512, 8, 2, 64

    source = _DummyAttention(hidden, num_heads, num_kv_heads, head_dim)
    source = source.to(device).to(torch.bfloat16)

    tinker_mod = TinKerAttention.from_module(source).to(device=device, dtype=torch.bfloat16)

    assert tinker_mod.num_heads == num_heads
    assert tinker_mod.num_kv_heads == num_kv_heads
    assert tinker_mod.head_dim == head_dim

    torch.testing.assert_close(tinker_mod.q_proj.weight.data, source.q_proj.weight.data)
    torch.testing.assert_close(tinker_mod.k_proj.weight.data, source.k_proj.weight.data)
    torch.testing.assert_close(tinker_mod.v_proj.weight.data, source.v_proj.weight.data)
    torch.testing.assert_close(tinker_mod.o_proj.weight.data, source.o_proj.weight.data)
