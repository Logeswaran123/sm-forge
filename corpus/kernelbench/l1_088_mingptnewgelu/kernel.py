import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    k: tl.constexpr = 0.7978845608028654  # sqrt(2/pi)
    u = k * (x + 0.044715 * x * x * x)
    # tanh(u) = 1 - 2 / (exp(2u) + 1), stable for large |u|
    t = 1.0 - 2.0 / (tl.exp(2.0 * u) + 1.0)
    y = 0.5 * x * (1.0 + t)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        xc = x.contiguous()
        y = torch.empty_like(xc)
        n = xc.numel()
        if n == 0:
            return y.view(x.shape)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _gelu_tanh_kernel[grid](xc.view(-1), y.view(-1), n, BLOCK=BLOCK, num_warps=4)
        return y.view(x.shape)
