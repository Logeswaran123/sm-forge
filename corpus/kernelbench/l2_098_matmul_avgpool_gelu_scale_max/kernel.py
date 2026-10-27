import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemm_pool_gelu_max_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    M, N, K, N_OUT,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    POOL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Average pool over groups of POOL along N
    G: tl.constexpr = BLOCK_N // POOL
    acc3 = tl.reshape(acc, (BLOCK_M, G, POOL))
    pooled = tl.sum(acc3, axis=2) / POOL  # [BLOCK_M, G]

    # Exact GELU
    gl = 0.5 * pooled * (1.0 + tl.math.erf(pooled * 0.7071067811865476))
    gl = gl * scale

    offs_g = pid_n * G + tl.arange(0, G)
    g_valid = offs_g < N_OUT
    gl = tl.where(g_valid[None, :], gl, float("-inf"))

    row_max = tl.max(gl, axis=1)  # [BLOCK_M]
    tl.store(part_ptr + pid_n * M + offs_m, row_max, mask=m_mask)


@triton.jit
def _reduce_max_kernel(
    part_ptr, out_ptr, M, NB,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    res = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    for j in range(0, NB):
        v = tl.load(part_ptr + j * M + offs_m, mask=m_mask, other=float("-inf"))
        res = tl.maximum(res, v)
    tl.store(out_ptr + offs_m, res.to(out_ptr.dtype.element_ty), mask=m_mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        w = self.matmul.weight
        b = self.matmul.bias
        M, K = x.shape
        N = w.shape[0]
        POOL = int(self.pool_kernel_size)
        N_OUT = N // POOL

        BLOCK_M = 128
        BLOCK_K = 32
        BLOCK_N = 128
        if BLOCK_N % POOL != 0:
            BLOCK_N = triton.next_power_of_2(POOL)
            BLOCK_N = max(BLOCK_N, 16)
        if POOL & (POOL - 1) != 0:
            raise NotImplementedError("pool size must be a power of 2")

        # Only columns belonging to complete pool groups matter
        N_eff = N_OUT * POOL
        NB = triton.cdiv(N_eff, BLOCK_N)

        part = torch.empty((NB, M), device=x.device, dtype=torch.float32)
        out = torch.empty((M,), device=x.device, dtype=x.dtype)

        grid = (triton.cdiv(M, BLOCK_M), NB)
        _gemm_pool_gelu_max_kernel[grid](
            x, w, b, part,
            M, N_eff, K, N_OUT,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            float(self.scale_factor),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            POOL=POOL,
            num_warps=8, num_stages=4,
        )

        RB = 128
        _reduce_max_kernel[(triton.cdiv(M, RB),)](
            part, out, M, NB, BLOCK_M=RB, num_warps=4,
        )
        return out
