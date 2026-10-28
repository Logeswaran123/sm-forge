import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_dim_kernel(
    x_ptr, out_ptr,
    R, INNER,
    stride_outer, stride_r,
    inv_r,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < INNER
    offs_r = tl.arange(0, BLOCK_R)

    base = x_ptr + pid_o.to(tl.int64) * stride_outer
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for r0 in range(0, R, BLOCK_R):
        rr = r0 + offs_r
        mask_r = rr < R
        ptrs = base + rr.to(tl.int64)[:, None] * stride_r + offs_n[None, :]
        m = mask_r[:, None] & mask_n[None, :]
        vals = tl.load(ptrs, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    res = acc * inv_r
    out_ptrs = out_ptr + pid_o.to(tl.int64) * INNER + offs_n
    tl.store(out_ptrs, res.to(out_ptr.dtype.element_ty), mask=mask_n)


class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim % x.dim()
        shape = list(x.shape)
        outer = 1
        for s in shape[:dim]:
            outer *= s
        R = shape[dim]
        inner = 1
        for s in shape[dim + 1:]:
            inner *= s
        out_shape = shape[:dim] + shape[dim + 1:]

        xc = x.contiguous().view(outer, R, inner)
        out = torch.empty((outer, inner), dtype=x.dtype, device=x.device)

        if outer == 0 or inner == 0:
            return out.view(out_shape)

        BLOCK_N = 1024
        BLOCK_R = 8
        grid = (outer, triton.cdiv(inner, BLOCK_N))
        inv_r = 1.0 / R if R > 0 else float("nan")
        mean_dim_kernel[grid](
            xc, out,
            R, inner,
            R * inner, inner,
            inv_r,
            BLOCK_R=BLOCK_R,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out.view(out_shape)
