"""fused rope + gqa attention with pytorch fallback."""

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# --- rope kernel ---

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_S": 32, "BLOCK_D": 32}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_S": 64, "BLOCK_D": 32}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_S": 32, "BLOCK_D": 64}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_S": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_S": 128, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        ],
        key=["seq_len", "half_dim"],
    )
    @triton.jit
    def _rope_kernel(
        X, COS, SIN,
        seq_len,
        half_dim: tl.constexpr,
        stride_b, stride_h, stride_s, stride_d,
        cos_stride_s,
        num_heads,
        start_pos,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        for s_start in range(0, seq_len, BLOCK_S):
            s_offsets = s_start + tl.arange(0, BLOCK_S)
            s_mask = s_offsets < seq_len

            for d_start in range(0, half_dim, BLOCK_D):
                d_offsets = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offsets < half_dim
                mask = s_mask[:, None] & d_mask[None, :]

                base = pid_b * stride_b + pid_h * stride_h
                even_idx = base + s_offsets[:, None] * stride_s + (d_offsets[None, :] * 2) * stride_d
                odd_idx = base + s_offsets[:, None] * stride_s + (d_offsets[None, :] * 2 + 1) * stride_d

                x_even = tl.load(X + even_idx, mask=mask, other=0.0).to(tl.float32)
                x_odd = tl.load(X + odd_idx, mask=mask, other=0.0).to(tl.float32)

                pos_offsets = (s_offsets + start_pos)[:, None]
                cos_idx = pos_offsets * cos_stride_s + d_offsets[None, :]
                cos_val = tl.load(COS + cos_idx, mask=mask, other=1.0).to(tl.float32)
                sin_val = tl.load(SIN + cos_idx, mask=mask, other=0.0).to(tl.float32)

                new_even = x_even * cos_val - x_odd * sin_val
                new_odd = x_even * sin_val + x_odd * cos_val

                tl.store(X + even_idx, new_even, mask=mask)
                tl.store(X + odd_idx, new_odd, mask=mask)


# --- attention kernel ---

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_S": 32, "BLOCK_KV": 32}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_S": 64, "BLOCK_KV": 32}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_S": 32, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_S": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_S": 128, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        ],
        key=["seq_len", "head_dim"],
    )
    @triton.jit
    def _attention_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        seq_len,
        head_dim: tl.constexpr,
        scale,
        num_kv_groups,
        BLOCK_S: tl.constexpr,
        BLOCK_KV: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_s = tl.program_id(2)

        # gqa head mapping
        kv_head = pid_h // num_kv_groups

        q_start = pid_s * BLOCK_S
        q_offsets = q_start + tl.arange(0, BLOCK_S)
        q_mask = q_offsets < seq_len
        d_offsets = tl.arange(0, head_dim)

        q_ptrs = (pid_b * stride_qb + pid_h * stride_qh +
                  q_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd)
        q = tl.load(Q + q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        # online softmax state
        m_i = tl.full([BLOCK_S], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_S], dtype=tl.float32)
        acc = tl.zeros([BLOCK_S, head_dim], dtype=tl.float32)

        # iterate kv blocks
        max_kv = tl.minimum(q_start + BLOCK_S, seq_len)
        for kv_start in range(0, max_kv, BLOCK_KV):
            kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = kv_offsets < seq_len

            k_ptrs = (pid_b * stride_kb + kv_head * stride_kh +
                      kv_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd)
            k = tl.load(K + k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

            qk = tl.dot(q, tl.trans(k)) * scale

            causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
            combined_mask = causal_mask & q_mask[:, None] & kv_mask[None, :]
            qk = tl.where(combined_mask, qk, float("-inf"))

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            acc = acc * alpha[:, None]
            v_ptrs = (pid_b * stride_vb + kv_head * stride_vh +
                      kv_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd)
            v = tl.load(V + v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
            acc += tl.dot(p.to(tl.float32), v)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        acc = acc / l_i[:, None]

        o_ptrs = (pid_b * stride_ob + pid_h * stride_oh +
                  q_offsets[:, None] * stride_os + d_offsets[None, :] * stride_od)
        tl.store(Out + o_ptrs, acc, mask=q_mask[:, None])


def precompute_rope_tables(head_dim, max_seq_len, theta=10000.0, device=None, dtype=torch.float32):
    """precompute sin/cos tables for rope."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _apply_rope_pytorch(x, cos_table, sin_table, start_pos=0):
    S = x.shape[2]
    cos = cos_table[start_pos:start_pos + S].unsqueeze(0).unsqueeze(0)
    sin = sin_table[start_pos:start_pos + S].unsqueeze(0).unsqueeze(0)
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()
    new_even = x_even * cos - x_odd * sin
    new_odd = x_even * sin + x_odd * cos
    return torch.stack([new_even, new_odd], dim=-1).flatten(-2).to(x.dtype)


def _attention_pytorch(q, k, v, cos_table, sin_table, num_heads, num_kv_heads, head_dim, start_pos=0):
    B, S, _ = q.shape
    num_kv_groups = num_heads // num_kv_heads

    q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

    q = _apply_rope_pytorch(q, cos_table, sin_table, start_pos)
    k = _apply_rope_pytorch(k, cos_table, sin_table, start_pos)

    if num_kv_groups > 1:
        k = k.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, num_heads, S, head_dim)
        v = v.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, num_heads, S, head_dim)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).contiguous().view(B, S, num_heads * head_dim)


def _attention_triton(q, k, v, cos_table, sin_table, num_heads, num_kv_heads, head_dim, start_pos=0):
    B, S, _ = q.shape
    num_kv_groups = num_heads // num_kv_heads

    q = q.view(B, S, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    k = k.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    v = v.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    _rope_kernel[(B, num_heads)](
        q, cos_table, sin_table,
        S, head_dim // 2,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cos_table.stride(0), num_heads, start_pos,
    )
    _rope_kernel[(B, num_kv_heads)](
        k, cos_table, sin_table,
        S, head_dim // 2,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos_table.stride(0), num_kv_heads, start_pos,
    )

    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)

    def grid(meta):
        return (B, num_heads, triton.cdiv(S, meta["BLOCK_S"]))

    _attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S, head_dim, scale, num_kv_groups,
    )

    return out.permute(0, 2, 1, 3).contiguous().view(B, S, num_heads * head_dim)


def fused_attention(q, k, v, cos_table, sin_table, num_heads, num_kv_heads, head_dim, start_pos=0):
    """fused rope + gqa + attention."""
    if HAS_TRITON and q.is_cuda:
        return _attention_triton(q, k, v, cos_table, sin_table, num_heads, num_kv_heads, head_dim, start_pos)
    return _attention_pytorch(q, k, v, cos_table, sin_table, num_heads, num_kv_heads, head_dim, start_pos)
