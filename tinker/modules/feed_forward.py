"""fused swiglu module"""

import torch
import torch.nn as nn

from tinker.kernels.fused_swiglu import fused_swiglu


class TinKerSwiGLU(nn.Module):
    """fused swiglu ffn

    args:
        hidden_size: hidden dim
        intermediate_size: intermediate dim
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        hidden = fused_swiglu(gate_out, up_out)
        return self.down_proj(hidden)

    @classmethod
    def from_module(cls, source: nn.Module) -> "TinKerSwiGLU":
        """from existing module"""
        gate_w, up_w, down_w = _find_swiglu_weights(source)

        hidden_size = gate_w.shape[1]
        intermediate_size = gate_w.shape[0]

        module = cls(hidden_size=hidden_size, intermediate_size=intermediate_size)
        module.gate_proj.weight.data.copy_(gate_w.data)
        module.up_proj.weight.data.copy_(up_w.data)
        module.down_proj.weight.data.copy_(down_w.data)
        return module

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"


def _find_swiglu_weights(module: nn.Module):
    """find projection weights"""
    gate_names = ["gate_proj", "w1", "gate"]
    up_names = ["up_proj", "w3", "up"]
    down_names = ["down_proj", "w2", "down"]

    def _get_weight(names):
        for name in names:
            sub = getattr(module, name, None)
            if sub is not None:
                if hasattr(sub, "weight"):
                    return sub.weight
                if isinstance(sub, torch.Tensor):
                    return sub
        return None

    gate_w = _get_weight(gate_names)
    up_w = _get_weight(up_names)
    down_w = _get_weight(down_names)

    if gate_w is None or up_w is None or down_w is None:
        raise ValueError(
            f"Could not find SwiGLU projection weights in {type(module).__name__}. "
            f"Expected attributes named one of: gate_proj/w1/gate, up_proj/w3/up, down_proj/w2/down"
        )

    return gate_w, up_w, down_w
