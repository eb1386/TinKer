"""swiglu tests"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


# reference implementation

def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """pytorch reference"""
    return F.silu(gate) * up


# kernel tests

KERNEL_CONFIGS = [
    pytest.param(688, 1376, 64, 1, id="inter1376-seq64"),
    pytest.param(1024, 2816, 64, 1, id="inter2816-seq64"),
    pytest.param(1280, 3584, 64, 1, id="inter3584-seq64"),
    pytest.param(1024, 4096, 64, 1, id="inter4096-seq64"),
    pytest.param(512, 1376, 1, 1, id="edge-seq1"),
    pytest.param(1024, 2816, 32, 8, id="large-batch8"),
]


@skip_no_cuda
@pytest.mark.parametrize("hidden_size, intermediate_size, seq_len, batch", KERNEL_CONFIGS)
def test_fused_swiglu_kernel(hidden_size, intermediate_size, seq_len, batch):
    """kernel correctness"""
    from tinker.kernels.fused_swiglu import fused_swiglu

    torch.manual_seed(42)
    device = torch.device("cuda")

    gate = torch.randn(batch, seq_len, intermediate_size, device=device, dtype=torch.bfloat16)
    up = torch.randn(batch, seq_len, intermediate_size, device=device, dtype=torch.bfloat16)

    expected = swiglu_ref(gate, up)
    actual = fused_swiglu(gate, up)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


# module tests

@skip_no_cuda
@pytest.mark.parametrize("hidden_size, intermediate_size, seq_len, batch", KERNEL_CONFIGS)
def test_tinker_swiglu_module(hidden_size, intermediate_size, seq_len, batch):
    """module correctness"""
    from tinker.modules import TinKerSwiGLU

    torch.manual_seed(42)
    device = torch.device("cuda")

    module = TinKerSwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)
    module = module.to(device).to(torch.bfloat16)

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=torch.bfloat16)

    gate_out = module.gate_proj(x)
    up_out = module.up_proj(x)
    expected_hidden = swiglu_ref(gate_out, up_out)
    expected = module.down_proj(expected_hidden)

    actual = module(x)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


# from_module test

class _DummySwiGLU(nn.Module):
    """dummy swiglu"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@skip_no_cuda
def test_from_module():
    """from_module correctness"""
    from tinker.modules import TinKerSwiGLU

    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden_size = 512
    intermediate_size = 1376

    source = _DummySwiGLU(hidden_size, intermediate_size).to(device).to(torch.bfloat16)

    tinker_mod = TinKerSwiGLU.from_module(source).to(device=device, dtype=torch.bfloat16)

    # verify weights
    assert tinker_mod.hidden_size == hidden_size
    assert tinker_mod.intermediate_size == intermediate_size
    torch.testing.assert_close(tinker_mod.gate_proj.weight.data, source.gate_proj.weight.data)
    torch.testing.assert_close(tinker_mod.up_proj.weight.data, source.up_proj.weight.data)
    torch.testing.assert_close(tinker_mod.down_proj.weight.data, source.down_proj.weight.data)

    x = torch.randn(1, 16, hidden_size, device=device, dtype=torch.bfloat16)
    expected = source(x)
    actual = tinker_mod(x)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
