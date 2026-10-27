import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_row_kernel(
    x_ptr, y_ptr,
    n_cols,
    stride_xr, stride_yr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row.to(tl.int64) * stride_xr
    y_row = y_ptr + row.to(tl.int64) * stride_yr

    offs = tl.arange(0, BLOCK)
    NEG: tl.constexpr = -1.0e30

    m = tl.full([BLOCK], NEG, dtype=tl.float32)
    s = tl.zeros([BLOCK], dtype=tl.float32)

    # Pass 1: per-lane online max / sum
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=NEG).to(tl.float32)
        new_m = tl.maximum(m, x)
        e = tl.where(mask, tl.exp(x - new_m), 0.0)
        s = s * tl.exp(m - new_m) + e
        m = new_m

    row_max = tl.max(m, axis=0)
    row_sum = tl.sum(s * tl.exp(m - row_max), axis=0)
    inv = 1.0 / row_sum

    # Pass 2: write normalized output
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=NEG).to(tl.float32)
        y = tl.exp(x - row_max) * inv
        tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    """
    Softmax over dim=1 implemented with a Triton kernel.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2
        if x.stride(1) != 1:
            x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)
        if n_rows == 0 or n_cols == 0:
            return y
        BLOCK = 4096
        _softmax_row_kernel[(n_rows,)](
            x, y,
            n_cols,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
