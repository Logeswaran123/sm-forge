import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _log_softmax_kernel(x_ptr, y_ptr, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_y

    offs0 = tl.arange(0, BLOCK)
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)

    # Pass 1: per-lane online max / sum-exp
    for start in range(0, N, BLOCK):
        offs = start + offs0
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        valid = m_new != float("-inf")
        alpha = tl.where(valid, tl.exp(m - m_new), 0.0)
        p = tl.where(valid, tl.exp(x - m_new), 0.0)
        s = s * alpha + p
        m = m_new

    M = tl.max(m, axis=0)
    scale = tl.where(m != float("-inf"), tl.exp(m - M), 0.0)
    total = tl.sum(s * scale, axis=0)
    lse = M + tl.log(total)

    # Pass 2: write x - lse
    for start in range(0, N, BLOCK):
        offs = start + offs0
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - lse
        tl.store(y_row + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim % x.dim()
        moved = dim != x.dim() - 1
        xt = x.movedim(dim, -1) if moved else x
        xt = xt.contiguous()
        shape = xt.shape
        N = shape[-1]
        x2 = xt.view(-1, N)
        R = x2.shape[0]
        y2 = torch.empty_like(x2)
        if R > 0 and N > 0:
            _log_softmax_kernel[(R,)](
                x2, y2, N, x2.stride(0), y2.stride(0),
                BLOCK=4096, num_warps=8,
            )
        y = y2.view(shape)
        if moved:
            y = y.movedim(-1, dim)
        return y
