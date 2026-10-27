import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_pool_sum_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    M, N, K, N_VALID,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pb, stride_pm,
    POOL_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :].to(tl.int64) * stride_xk
    w_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None].to(tl.int64) * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        xt = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        wt = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        wt = wt.to(xt.dtype)
        acc = tl.dot(xt, wt, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # columns beyond the last full pooling window are dropped (MaxPool1d semantics)
    valid_n = offs_n < N_VALID
    acc = tl.where(valid_n[None, :], acc, float("-inf"))

    pooled = tl.max(tl.reshape(acc, (BLOCK_M, BLOCK_N // POOL_K, POOL_K)), axis=2)
    pooled = tl.where(pooled == float("-inf"), 0.0, pooled)
    row_sum = tl.sum(pooled, axis=1)  # (BLOCK_M,)

    tl.store(part_ptr + pid_n * stride_pb + offs_m * stride_pm, row_sum, mask=m_mask)


@triton.jit
def reduce_scale_kernel(
    part_ptr, out_ptr, M, NB, scale,
    stride_pb, stride_pm,
    BLOCK_M: tl.constexpr, BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_b = tl.arange(0, BLOCK_B)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for b0 in range(0, NB, BLOCK_B):
        bm = (b0 + offs_b) < NB
        p = tl.load(
            part_ptr + (b0 + offs_b)[:, None] * stride_pb + offs_m[None, :] * stride_pm,
            mask=bm[:, None] & m_mask[None, :], other=0.0,
        )
        acc += tl.sum(p, axis=0)
    acc = acc * scale
    tl.store(out_ptr + offs_m, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size

    def forward(self, x):
        assert x.dim() == 2
        ks = self.kernel_size
        assert ks >= 1 and (ks & (ks - 1)) == 0, "pool kernel size must be a power of 2"
        x = x.contiguous()
        w = self.matmul.weight
        b = self.matmul.bias
        M, K = x.shape
        N = w.shape[0]
        n_valid = (N // ks) * ks

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
        assert BLOCK_N % ks == 0
        nb = triton.cdiv(N, BLOCK_N)
        partials = torch.empty((nb, M), device=x.device, dtype=torch.float32)

        grid = (nb, triton.cdiv(M, BLOCK_M))
        fused_linear_pool_sum_kernel[grid](
            x, w, b, partials,
            M, N, K, n_valid,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            partials.stride(0), partials.stride(1),
            POOL_K=ks,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        out = torch.empty((M,), device=x.device, dtype=x.dtype)
        RB_M, RB_B = 32, 64
        reduce_scale_kernel[(triton.cdiv(M, RB_M),)](
            partials, out, M, nb, float(self.scale_factor),
            partials.stride(0), partials.stride(1),
            BLOCK_M=RB_M, BLOCK_B=RB_B,
        )
        return out
