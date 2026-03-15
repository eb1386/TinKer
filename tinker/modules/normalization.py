"""fused rmsnorm module"""

import torch
import torch.nn as nn

from tinker.kernels.fused_rmsnorm import fused_rmsnorm


class TinKerRMSNorm(nn.Module):
    """fused rmsnorm layer

    args:
        dim: hidden dim
        eps: epsilon
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)

    @classmethod
    def from_module(cls, source: nn.Module) -> "TinKerRMSNorm":
        """from existing module"""
        weight = source.weight
        dim = weight.shape[0]
        eps = getattr(source, "eps", getattr(source, "variance_epsilon", 1e-6))

        module = cls(dim=dim, eps=eps)
        module.weight.data.copy_(weight.data)
        return module

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
