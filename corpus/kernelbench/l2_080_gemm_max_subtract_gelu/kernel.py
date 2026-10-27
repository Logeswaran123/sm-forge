import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemm_rowmax_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None], other=0.0,
        )
        acc = tl.dot(a.to(b.dtype), b, acc)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.where(n_mask[None, :], acc, float("-inf"))
    row_max = tl.max(acc, axis=1)
    tl.store(part_ptr + offs_m * stride_pm + pid_n, row_max, mask=m_mask)


@triton.jit
def _finalize_kernel(
    part_ptr, out_ptr,
    M, T,
    stride_pm, stride_om,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_T)
    vals = tl.load(part_ptr + row * stride_pm + offs, mask=offs < T, other=float("-inf"))
    m = tl.max(vals, axis=0)
    # round to output dtype like the reference (linear output is in that dtype)
    m = m.to(out_ptr.dtype.element_ty).to(tl.float32)
    mean = m  # mean over a dim of size 1
    v = m - mean
    g = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
    tl.store(out_ptr + row * stride_om, g.to(out_ptr.dtype.element_ty))


class Model(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        assert self.max_dim == 1
        w = self.gemm.weight
        b = self.gemm.bias
        M, K = x.shape
        N = w.shape[0]

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        T = triton.cdiv(N, BLOCK_N)
        part = torch.empty((M, T), device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(M, BLOCK_M), T)
        _gemm_rowmax_kernel[grid](
            x, w, b, part,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            part.stride(0),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        BLOCK_T = max(triton.next_power_of_2(T), 16)
        _finalize_kernel[(M,)](
            part, out, M, T,
            part.stride(0), out.stride(0),
            BLOCK_T=BLOCK_T,
            num_warps=1,
        )
        return out
