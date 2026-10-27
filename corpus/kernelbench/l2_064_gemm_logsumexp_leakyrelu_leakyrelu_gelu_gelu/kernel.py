import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_lse_partial_kernel(
    x_ptr, w_ptr, b_ptr, pm_ptr, ps_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
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
        acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    acc = tl.where(n_mask[None, :], acc, float("-inf"))
    row_max = tl.max(acc, axis=1)
    p = tl.exp(acc - row_max[:, None])
    p = tl.where(n_mask[None, :], p, 0.0)
    row_sum = tl.sum(p, axis=1)

    tl.store(pm_ptr + offs_m * stride_pm + pid_n, row_max, mask=m_mask)
    tl.store(ps_ptr + offs_m * stride_pm + pid_n, row_sum, mask=m_mask)


@triton.jit
def _gelu(v):
    return 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))


@triton.jit
def lse_finish_kernel(
    pm_ptr, ps_ptr, out_ptr,
    NB, stride_pm,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < NB
    pm = tl.load(pm_ptr + row * stride_pm + offs, mask=mask, other=float("-inf"))
    ps = tl.load(ps_ptr + row * stride_pm + offs, mask=mask, other=0.0)
    m = tl.max(pm, axis=0)
    s = tl.sum(tl.where(mask, ps * tl.exp(pm - m), 0.0), axis=0)
    v = m + tl.log(s)
    # LeakyReLU x2
    v = tl.where(v > 0, v, v * 0.01)
    v = tl.where(v > 0, v, v * 0.01)
    # GELU x2 (exact erf form)
    v = _gelu(v)
    v = _gelu(v)
    tl.store(out_ptr + row, v.to(out_ptr.dtype.element_ty))


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        w = self.linear.weight
        b = self.linear.bias
        M, K = x.shape
        N = w.shape[0]

        BM, BN, BK = 64, 128, 64
        nb = triton.cdiv(N, BN)
        pm = torch.empty((M, nb), device=x.device, dtype=torch.float32)
        ps = torch.empty((M, nb), device=x.device, dtype=torch.float32)

        has_bias = b is not None
        b_arg = b if has_bias else w

        grid = (triton.cdiv(M, BM), nb)
        gemm_lse_partial_kernel[grid](
            x, w, b_arg, pm, ps,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            pm.stride(0),
            HAS_BIAS=has_bias,
            BM=BM, BN=BN, BK=BK,
            num_warps=4, num_stages=3,
        )

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        BLOCK = max(triton.next_power_of_2(nb), 16)
        lse_finish_kernel[(M,)](
            pm, ps, out,
            nb, pm.stride(0),
            BLOCK=BLOCK,
            num_warps=1,
        )
        return out
