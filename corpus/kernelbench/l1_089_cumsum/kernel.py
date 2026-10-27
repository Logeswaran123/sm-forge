import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _row_cumsum_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    offs = tl.arange(0, BLOCK)
    carry = tl.zeros((), dtype=tl.float32)
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        c = tl.cumsum(v, axis=0) + carry
        tl.store(y_ptr + base + idx, c.to(y_ptr.dtype.element_ty), mask=mask)
        carry += tl.sum(v, axis=0)


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        nd = x.dim()
        d = self.dim % nd
        if d != nd - 1:
            perm = [i for i in range(nd) if i != d] + [d]
            xt = x.permute(*perm).contiguous()
        else:
            perm = None
            xt = x.contiguous()
        N = xt.shape[-1]
        M = xt.numel() // N if N > 0 else 0
        y = torch.empty_like(xt)
        if M > 0 and N > 0:
            BLOCK = 1024
            _row_cumsum_kernel[(M,)](xt, y, N, BLOCK=BLOCK, num_warps=4)
        if perm is not None:
            inv = [0] * nd
            for i, p in enumerate(perm):
                inv[p] = i
            y = y.permute(*inv)
        return y
