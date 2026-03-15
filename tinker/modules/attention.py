"""fused attention module"""

import torch
import torch.nn as nn

from tinker.kernels.fused_attention import fused_attention, precompute_rope_tables


class TinKerAttention(nn.Module):
    """fused rope+gqa attention

    args:
        hidden_size: hidden dim
        num_heads: query heads
        num_kv_heads: kv heads
        head_dim: per-head dim
        rope_theta: rope base
        max_seq_len: max length
        dropout: attn dropout
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        cos_table, sin_table = precompute_rope_tables(
            self.head_dim, max_seq_len, rope_theta
        )
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """forward pass"""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn_out = fused_attention(
            q, k, v,
            self.cos_table, self.sin_table,
            self.num_heads, self.num_kv_heads, self.head_dim,
            start_pos=start_pos,
        )

        return self.o_proj(attn_out)

    @classmethod
    def from_module(
        cls,
        source: nn.Module,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ) -> "TinKerAttention":
        """from existing module"""
        q_proj, k_proj, v_proj, o_proj = _find_attention_projections(source)

        hidden_size = q_proj.weight.shape[1]
        q_out = q_proj.weight.shape[0]
        kv_out = k_proj.weight.shape[0]

        # infer head counts
        head_dim = getattr(source, "head_dim", None)
        if head_dim is None:
            # try common dims
            for candidate in [64, 128]:
                if q_out % candidate == 0 and kv_out % candidate == 0:
                    head_dim = candidate
                    break
            if head_dim is None:
                head_dim = hidden_size // (q_out // (kv_out // 1))

        num_heads = q_out // head_dim
        num_kv_heads = kv_out // head_dim

        # source rope_theta
        source_theta = getattr(source, "rope_theta", None)
        if source_theta is not None:
            rope_theta = source_theta

        module = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )

        module.q_proj.weight.data.copy_(q_proj.weight.data)
        module.k_proj.weight.data.copy_(k_proj.weight.data)
        module.v_proj.weight.data.copy_(v_proj.weight.data)
        module.o_proj.weight.data.copy_(o_proj.weight.data)

        device = q_proj.weight.device
        module.cos_table = module.cos_table.to(device)
        module.sin_table = module.sin_table.to(device)

        return module

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"rope_theta={self.rope_theta}"
        )


def _find_attention_projections(module: nn.Module):
    """find projection layers"""
    q_names = ["q_proj", "query", "wq", "q"]
    k_names = ["k_proj", "key", "wk", "k"]
    v_names = ["v_proj", "value", "wv", "v"]
    o_names = ["o_proj", "output", "wo", "out_proj", "o"]

    def _get_proj(names):
        for name in names:
            sub = getattr(module, name, None)
            if sub is not None and hasattr(sub, "weight"):
                return sub
        return None

    q = _get_proj(q_names)
    k = _get_proj(k_names)
    v = _get_proj(v_names)
    o = _get_proj(o_names)

    if q is None or k is None or v is None or o is None:
        raise ValueError(
            f"Could not find attention projection layers in {type(module).__name__}. "
            f"Expected attributes named one of: q_proj/query/wq, k_proj/key/wk, "
            f"v_proj/value/wv, o_proj/output/wo"
        )

    return q, k, v, o
