import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_min_sub_kernel(
    x_ptr, w_ptr, b_ptr, c_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
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
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
        a_mask = (offs_m[:, None] < M) & (kk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # W tile viewed as [BLOCK_K, BLOCK_N]
        b_ptrs = w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk
        b_mask = (offs_n[None, :] < N) & (kk[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        b = b.to(a.dtype)
        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    c = tl.load(c_ptr).to(tl.float32)
    acc = tl.minimum(acc, c) - c

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        w = self.linear.weight
        b = self.linear.bias
        c = self.constant
        orig_shape = x.shape
        K = orig_shape[-1]
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M = 16 if M <= 16 else 64
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_min_sub_kernel[grid](
            x2, w, b, c, out,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return out.reshape(*orig_shape[:-1], N)
