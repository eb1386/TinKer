"""fused swiglu kernel with pytorch fallback."""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=2, num_stages=2),
        ],
        key=["N"],
    )
    @triton.jit
    def _swiglu_kernel(
        Gate, Up, Out,
        N: tl.constexpr,
        stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        Gate += row * stride
        Up += row * stride
        Out += row * stride

        for offset in range(0, N, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            gate = tl.load(Gate + cols, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(Up + cols, mask=mask, other=0.0).to(tl.float32)
            silu_gate = gate * tl.sigmoid(gate)
            out = silu_gate * up
            tl.store(Out + cols, out, mask=mask)


def _swiglu_triton(gate, up):
    orig_shape = gate.shape
    gate_2d = gate.reshape(-1, orig_shape[-1])
    up_2d = up.reshape(-1, orig_shape[-1])
    num_rows, N = gate_2d.shape
    out = torch.empty_like(gate_2d)
    _swiglu_kernel[(num_rows,)](gate_2d, up_2d, out, N, gate_2d.stride(0))
    return out.reshape(orig_shape)


def _swiglu_pytorch(gate, up):
    return F.silu(gate).mul_(up)


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """apply fused swiglu activation."""
    assert gate.shape == up.shape, "shape mismatch"
    if HAS_TRITON and gate.is_cuda:
        return _swiglu_triton(gate, up)
    return _swiglu_pytorch(gate, up)
