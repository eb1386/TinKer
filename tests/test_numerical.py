"""numerical precision tests"""

import math

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


# rmsnorm precision

def test_rmsnorm_float32_precision():
    """float32 precision"""
    from tinker.kernels.fused_rmsnorm import fused_rmsnorm

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 1024
    seq_len = 64
    batch = 2

    x_bf16 = torch.randn(batch, seq_len, dim, device=device, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(dim, device=device, dtype=torch.bfloat16)

    # float32 reference
    x_f32 = x_bf16.float()
    w_f32 = weight_bf16.float()
    norm = x_f32.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    ref_f32 = (x_f32 * norm * w_f32)

    # kernel output
    kernel_out = fused_rmsnorm(x_bf16, weight_bf16, eps=1e-6)

    torch.testing.assert_close(
        kernel_out.float(),
        ref_f32.to(torch.bfloat16).float(),
        atol=1e-2,
        rtol=1e-2,
    )


# rope correctness

def test_rope_analytical():
    """rope analytical test"""
    from tinker.kernels.fused_attention import precompute_rope_tables, _apply_rope_pytorch

    device = torch.device("cpu")
    head_dim = 4
    seq_len = 2
    B = 1

    cos_table, sin_table = precompute_rope_tables(head_dim, max_seq_len=16, device=device)

    # ones input as (B, H, S, D)
    q_heads = torch.ones(B, 1, seq_len, head_dim, device=device, dtype=torch.float32)

    # apply rope
    q_rot = _apply_rope_pytorch(q_heads, cos_table, sin_table, start_pos=0)

    # position 0: cos=1, sin=0 -> identity
    torch.testing.assert_close(
        q_rot[0, 0, 0, :],
        torch.ones(head_dim, device=device),
        atol=1e-5,
        rtol=1e-5,
    )

    # position 1: compute expected
    cos_pos1 = cos_table[1, :].float()
    sin_pos1 = sin_table[1, :].float()
    expected_pos1 = torch.zeros(head_dim, device=device)
    expected_pos1[0::2] = cos_pos1 - sin_pos1
    expected_pos1[1::2] = sin_pos1 + cos_pos1

    torch.testing.assert_close(
        q_rot[0, 0, 1, :],
        expected_pos1,
        atol=1e-5,
        rtol=1e-5,
    )


# single token

def test_attention_single_token():
    """single token output"""
    from tinker.kernels.fused_attention import fused_attention, precompute_rope_tables

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_dim = 64
    num_heads = 4
    num_kv_heads = 2
    hidden = num_heads * head_dim
    B, S = 1, 1

    cos_table, sin_table = precompute_rope_tables(head_dim, max_seq_len=16, device=device)

    q = torch.randn(B, S, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, S, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, S, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)

    out = fused_attention(
        q.clone(), k.clone(), v.clone(),
        cos_table, sin_table,
        num_heads, num_kv_heads, head_dim,
    )

    # expand for gqa
    v_heads = v.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    num_kv_groups = num_heads // num_kv_heads
    v_expanded = v_heads.repeat_interleave(num_kv_groups, dim=1)
    expected = v_expanded.permute(0, 2, 1, 3).contiguous().view(B, S, num_heads * head_dim)

    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)


def test_attention_uniform_values():
    """uniform values test"""
    from tinker.kernels.fused_attention import fused_attention, precompute_rope_tables

    torch.manual_seed(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_dim = 64
    num_heads = 4
    num_kv_heads = 2
    B, S = 1, 16

    cos_table, sin_table = precompute_rope_tables(head_dim, max_seq_len=64, device=device)

    q = torch.randn(B, S, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, S, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)

    # uniform v
    v_single = torch.randn(1, 1, num_kv_heads * head_dim, device=device, dtype=torch.bfloat16)
    v = v_single.expand(B, S, -1).contiguous()

    out = fused_attention(
        q.clone(), k.clone(), v.clone(),
        cos_table, sin_table,
        num_heads, num_kv_heads, head_dim,
    )

    # expected output
    v_heads = v_single.view(1, 1, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    num_kv_groups = num_heads // num_kv_heads
    v_expanded = v_heads.repeat_interleave(num_kv_groups, dim=1)
    expected_row = v_expanded.permute(0, 2, 1, 3).contiguous().view(1, 1, num_heads * head_dim)
    expected = expected_row.expand(B, S, -1)

    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-1)
