"""rmsnorm tests"""

import pytest
import torch
import torch.nn as nn

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


# reference implementation

def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """pytorch reference"""
    norm = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * norm).type_as(x) * weight


# kernel tests

KERNEL_CONFIGS = [
    pytest.param(256, 32, 1, id="tiny-dim256-seq32"),
    pytest.param(1024, 128, 1, id="plasma1.0-dim1024-seq128"),
    pytest.param(1280, 128, 1, id="plasma1.1-dim1280-seq128"),
    pytest.param(512, 64, 1, id="mha-dim512-seq64"),
    pytest.param(1024, 1, 1, id="edge-seq1"),
    pytest.param(1024, 64, 8, id="large-batch8"),
]


@skip_no_cuda
@pytest.mark.parametrize("dim, seq_len, batch", KERNEL_CONFIGS)
def test_fused_rmsnorm_kernel(dim, seq_len, batch):
    """kernel correctness"""
    from tinker.kernels.fused_rmsnorm import fused_rmsnorm

    torch.manual_seed(42)
    device = torch.device("cuda")

    x = torch.randn(batch, seq_len, dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, device=device, dtype=torch.bfloat16)

    expected = rmsnorm_ref(x, weight)
    actual = fused_rmsnorm(x, weight)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


# module tests

@skip_no_cuda
@pytest.mark.parametrize("dim, seq_len, batch", KERNEL_CONFIGS)
def test_tinker_rmsnorm_module(dim, seq_len, batch):
    """module correctness"""
    from tinker.modules import TinKerRMSNorm

    torch.manual_seed(42)
    device = torch.device("cuda")

    module = TinKerRMSNorm(dim=dim, eps=1e-6).to(device).to(torch.bfloat16)
    x = torch.randn(batch, seq_len, dim, device=device, dtype=torch.bfloat16)

    expected = rmsnorm_ref(x, module.weight, eps=1e-6)
    actual = module(x)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


# from_module test

class _DummyRMSNorm(nn.Module):
    """dummy rmsnorm"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


@skip_no_cuda
def test_from_module():
    """from_module correctness"""
    from tinker.modules import TinKerRMSNorm

    torch.manual_seed(0)
    device = torch.device("cuda")
    dim = 512

    source = _DummyRMSNorm(dim, eps=1e-5).to(device).to(torch.bfloat16)
    nn.init.normal_(source.weight)

    tinker_mod = TinKerRMSNorm.from_module(source).to(device=device, dtype=torch.bfloat16)

    assert tinker_mod.eps == source.eps
    torch.testing.assert_close(tinker_mod.weight.data, source.weight.data)

    x = torch.randn(1, 32, dim, device=device, dtype=torch.bfloat16)
    expected = rmsnorm_ref(x, source.weight, eps=source.eps)
    actual = tinker_mod(x)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
