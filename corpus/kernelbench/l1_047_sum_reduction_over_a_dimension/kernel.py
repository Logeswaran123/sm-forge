import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sum_dim_kernel(
    x_ptr, out_ptr,
    R, INNER,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < INNER
    offs_r = tl.arange(0, BLOCK_R)

    base = pid_o.to(tl.int64) * R.to(tl.int64) * INNER.to(tl.int64)
    acc = tl.zeros((BLOCK_R, BLOCK_N), dtype=tl.float32)

    for r0 in range(0, R, BLOCK_R):
        rr = r0 + offs_r
        mask = (rr[:, None] < R) & mask_n[None, :]
        ptrs = x_ptr + base + rr[:, None].to(tl.int64) * INNER + offs_n[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += vals.to(tl.float32)

    total = tl.sum(acc, axis=0)
    out_off = pid_o.to(tl.int64) * INNER + offs_n
    tl.store(out_ptr + out_off, total.to(out_ptr.dtype.element_ty), mask=mask_n)


class Model(nn.Module):
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        if dim < 0:
            dim += x.dim()
        x = x.contiguous()
        shape = list(x.shape)
        outer = 1
        for s in shape[:dim]:
            outer *= s
        R = shape[dim]
        inner = 1
        for s in shape[dim + 1:]:
            inner *= s

        out_shape = shape.copy()
        out_shape[dim] = 1
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        if outer == 0 or inner == 0:
            return out

        BLOCK_R = 32
        BLOCK_N = 128
        grid = (outer, triton.cdiv(inner, BLOCK_N))
        _sum_dim_kernel[grid](
            x, out, R, inner,
            BLOCK_R=BLOCK_R, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
