import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    N, stride_x, stride_o, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row.to(tl.int64) * stride_x + offs, mask=mask, other=0.0)
    h = x.to(tl.float32)
    ms = tl.sum(h * h, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    h = h * r
    # round to input dtype like reference (h.to(x.dtype))
    h = h.to(x_ptr.dtype.element_ty).to(tl.float32)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = w * h
    tl.store(out_ptr + row.to(tl.int64) * stride_o + offs,
             y.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(hidden_size))
        self.eps = eps

    def forward(self, x):
        N = x.shape[-1]
        orig_shape = x.shape
        x2 = x.reshape(-1, N)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        M = x2.shape[0]
        w = self.weight
        if not w.is_contiguous():
            w = w.contiguous()
        out_dtype = torch.promote_types(w.dtype, x.dtype)
        out = torch.empty((M, N), device=x.device, dtype=out_dtype)
        if M == 0:
            return out.reshape(orig_shape)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
        _rmsnorm_kernel[(M,)](
            x2, w, out,
            N, x2.stride(0), out.stride(0), float(self.eps),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.reshape(orig_shape)
