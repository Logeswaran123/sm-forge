import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(
    x_ptr, r_ptr, w_ptr, out_ptr, res_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row.to(tl.int64) * N

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    s = (x + r).to(res_ptr.dtype.element_ty)
    tl.store(res_ptr + base + offs, s, mask=mask)

    h = s.to(tl.float32)
    h = tl.where(mask, h, 0.0)
    ms = tl.sum(h * h, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    hn = (h * inv).to(x_ptr.dtype.element_ty).to(tl.float32)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = w * hn
    tl.store(out_ptr + base + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(hidden_size))
        self.eps = eps

    def forward(self, x, residual):
        N = x.shape[-1]
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, N)
        r2 = residual.contiguous().view(-1, N)
        M = x2.shape[0]
        res_dtype = torch.promote_types(x.dtype, residual.dtype)
        out_dtype = torch.promote_types(self.weight.dtype, x.dtype)
        res_out = torch.empty((M, N), dtype=res_dtype, device=x.device)
        out = torch.empty((M, N), dtype=out_dtype, device=x.device)
        w = self.weight.contiguous()
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        if M > 0:
            _add_rmsnorm_kernel[(M,)](
                x2, r2, w, out, res_out,
                N, float(self.eps),
                BLOCK=BLOCK, num_warps=num_warps,
            )
        return out.view(orig_shape), res_out.view(orig_shape)
