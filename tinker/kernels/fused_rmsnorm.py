"""fused rmsnorm kernel with pytorch fallback."""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=2, num_stages=2),
        ],
        key=["N"],
    )
    @triton.jit
    def _rmsnorm_kernel(
        X, W, Y,
        stride,
        N: tl.constexpr,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        X += row * stride
        Y += row * stride

        mean_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for offset in range(0, N, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            mean_sq += x * x

        var = tl.sum(mean_sq, axis=0) / N
        rrms = 1.0 / tl.sqrt(var + eps)

        for offset in range(0, N, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            y = x * rrms * w
            tl.store(Y + cols, y, mask=mask)


def _rmsnorm_triton(x, weight, eps=1e-6):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    num_rows, N = x_2d.shape
    y = torch.empty_like(x_2d)
    _rmsnorm_kernel[(num_rows,)](x_2d, weight, y, x_2d.stride(0), N, eps)
    return y.reshape(orig_shape)


def _rmsnorm_pytorch(x, weight, eps=1e-6):
    x_fp32 = x.float()
    norm = x_fp32.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x_fp32 * norm * weight.float()).to(x.dtype)


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """apply fused rmsnorm."""
    if HAS_TRITON and x.is_cuda:
        return _rmsnorm_triton(x, weight, eps)
    return _rmsnorm_pytorch(x, weight, eps)
