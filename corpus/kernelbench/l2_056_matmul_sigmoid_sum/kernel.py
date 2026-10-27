import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_sigmoid_partial_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pt, stride_pm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is [N, K]; load a [BLOCK_K, BLOCK_N] tile of W^T
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_rem = K - k0
        a = tl.load(x_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)
        b = tl.load(w_ptrs, mask=n_mask[None, :] & (offs_k[:, None] < k_rem), other=0.0)
        b = b.to(a.dtype)
        acc += tl.dot(a, b)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    z = acc + bias[None, :]
    s = 1.0 / (1.0 + tl.exp(-z))
    s = tl.where(n_mask[None, :], s, 0.0)
    row_sum = tl.sum(s, axis=1)  # [BLOCK_M]

    tl.store(part_ptr + pid_n * stride_pt + offs_m * stride_pm, row_sum, mask=m_mask)


@triton.jit
def reduce_partials_kernel(
    part_ptr, out_ptr,
    M, T,
    stride_pt, stride_pm,
    stride_om,
    BLOCK_M: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_t = tl.arange(0, BLOCK_T)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for t0 in range(0, T, BLOCK_T):
        t = t0 + offs_t
        ptrs = part_ptr + t[:, None] * stride_pt + offs_m[None, :] * stride_pm
        vals = tl.load(ptrs, mask=(t[:, None] < T) & m_mask[None, :], other=0.0)
        acc += tl.sum(vals, axis=0)

    tl.store(out_ptr + offs_m * stride_om, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        w = self.linear.weight
        b = self.linear.bias
        M, K = x.shape
        N = w.shape[0]

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
        T = triton.cdiv(N, BLOCK_N)

        partial = torch.empty((T, M), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), T)
        linear_sigmoid_partial_kernel[grid](
            x, w, b, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            partial.stride(0), partial.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        RBM, RBT = 32, 64
        reduce_partials_kernel[(triton.cdiv(M, RBM),)](
            partial, out,
            M, T,
            partial.stride(0), partial.stride(1),
            out.stride(0),
            BLOCK_M=RBM, BLOCK_T=RBT,
            num_warps=4,
        )
        return out
