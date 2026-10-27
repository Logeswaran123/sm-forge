import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_relu_div_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    inv_divisor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (kk[None, :] < K)
        a = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # W is [N, K]; load a [BLOCK_K, BLOCK_N] tile of W^T
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (kk[:, None] < K)
        b = tl.load(w_ptrs, mask=w_mask, other=0.0)
        b = b.to(a.dtype)

        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    acc = acc * inv_divisor

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        w = self.linear.weight
        bias = self.linear.bias
        K = x.shape[-1]
        N = w.shape[0]
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_relu_div_kernel[grid](
            x2, w, bias, out,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            1.0 / float(self.divisor),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return out.reshape(*lead, N)
