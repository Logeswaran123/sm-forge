import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    y = 0.5 * xf * (1.0 + tl.erf(xf * 0.7071067811865476))
    tl.store(y_ptr + offs, y.to(x.dtype), mask=mask)


class Model(nn.Module):
    """
    Simple model that performs a GELU activation (Triton implementation).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        x_flat = x_c.view(-1)
        n = x_flat.numel()
        y = torch.empty_like(x_c)
        if n == 0:
            return y
        y_flat = y.view(-1)
        BLOCK = 128
        grid = (triton.cdiv(n, BLOCK),)
        gelu_kernel[grid](x_flat, y_flat, n, BLOCK=BLOCK, num_warps=4)
        return y


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
