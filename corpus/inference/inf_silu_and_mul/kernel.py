import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    x_ptr, out_ptr,
    H,
    stride_xr, stride_or,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1)
    cols = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H

    x_row = x_ptr + row * stride_xr
    g = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_row + H + cols, mask=mask, other=0.0).to(tl.float32)

    s = g / (1.0 + tl.exp(-g))
    y = s * u

    tl.store(out_ptr + row * stride_or + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        H = N // 2
        x2 = x.reshape(-1, N)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        M = x2.shape[0]
        out = torch.empty((M, H), dtype=x.dtype, device=x.device)
        if M == 0 or H == 0:
            return out.reshape(*orig_shape[:-1], H)
        BLOCK = 1024
        grid = (M, triton.cdiv(H, BLOCK))
        _swiglu_kernel[grid](
            x2, out, H,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out.reshape(*orig_shape[:-1], H)
