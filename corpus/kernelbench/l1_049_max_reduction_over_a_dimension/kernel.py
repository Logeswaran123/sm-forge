import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _max_reduce_kernel(
    x_ptr, out_ptr,
    R, INNER,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_o = tl.program_id(1).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < INNER
    offs_r = tl.arange(0, BLOCK_R)

    base = x_ptr + pid_o * R * INNER
    acc = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)

    for r0 in range(0, R, BLOCK_R):
        rr = r0 + offs_r
        mask_r = rr < R
        ptrs = base + rr.to(tl.int64)[:, None] * INNER + offs_n[None, :]
        m = mask_r[:, None] & mask_n[None, :]
        vals = tl.load(ptrs, mask=m, other=float("-inf")).to(tl.float32)
        acc = tl.maximum(acc, tl.max(vals, axis=0))

    out_ptrs = out_ptr + pid_o * INNER + offs_n
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_n)


class Model(nn.Module):
    """
    Max reduction over a specific dimension (Triton implementation).
    """
    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nd = x.dim()
        dim = self.dim if self.dim >= 0 else self.dim + nd
        shape = list(x.shape)

        outer = 1
        for s in shape[:dim]:
            outer *= s
        R = shape[dim]
        inner = 1
        for s in shape[dim + 1:]:
            inner *= s

        x_c = x.contiguous()
        out_shape = shape[:dim] + shape[dim + 1:]
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

        if outer == 0 or inner == 0:
            return out

        BLOCK_R = 32
        BLOCK_N = 256
        grid = (triton.cdiv(inner, BLOCK_N), outer)
        _max_reduce_kernel[grid](
            x_c, out, R, inner,
            BLOCK_R=BLOCK_R, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out


batch_size = 128
dim1 = 4096
dim2 = 4095


def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]


def get_init_inputs():
    return [1]
