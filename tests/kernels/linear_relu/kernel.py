import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_relu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (k + offs_k)[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + (k + offs_k)[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & ((k + offs_k)[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(x, w, acc)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = tl.maximum(acc + bias[None, :], 0.0)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1 / in_features**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        M, K = x.shape
        N = self.weight.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        linear_relu_kernel[grid](
            x,
            self.weight,
            self.bias,
            y,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            self.weight.stride(0),
            self.weight.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
        )
        return y


def get_inputs():
    return [torch.randn(256, 128)]


def get_init_inputs():
    return [128, 192]
