import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    stride_x, stride_y,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b
    tl.store(y_ptr + row * stride_y + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(hidden_size))
        self.bias = nn.Parameter(0.1 * torch.randn(hidden_size))
        self.eps = eps

    def forward(self, x):
        N = x.shape[-1]
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, N)
        M = x2.shape[0]
        y = torch.empty((M, N), dtype=x.dtype, device=x.device)
        w = self.weight.contiguous()
        b = self.bias.contiguous()
        if M == 0:
            return y.view(orig_shape)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 2048 else (8 if BLOCK_N <= 8192 else 16)
        _layernorm_fwd_kernel[(M,)](
            x2, w, b, y,
            x2.stride(0), y.stride(0),
            N, float(self.eps),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
