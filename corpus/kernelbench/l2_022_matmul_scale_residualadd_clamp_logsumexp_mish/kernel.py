import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemm_lse_partial_kernel(
    x_ptr, w_ptr, b_ptr, pm_ptr, ps_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    NB,
    mult, cmin, cmax,
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
    y = (acc + bias[None, :]) * mult
    y = tl.minimum(tl.maximum(y, cmin), cmax)
    y = tl.where(n_mask[None, :], y, float("-inf"))

    rmax = tl.max(y, axis=1)
    e = tl.exp(y - rmax[:, None])
    e = tl.where(n_mask[None, :], e, 0.0)
    rsum = tl.sum(e, axis=1)

    tl.store(pm_ptr + offs_m * NB + pid_n, rmax, mask=m_mask)
    tl.store(ps_ptr + offs_m * NB + pid_n, rsum, mask=m_mask)


@triton.jit
def _lse_mish_kernel(
    pm_ptr, ps_ptr, out_ptr, NB, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    gmax = tl.full((BLOCK,), float("-inf"), tl.float32)
    for s in range(0, NB, BLOCK):
        idx = s + offs
        msk = idx < NB
        v = tl.load(pm_ptr + row * NB + idx, mask=msk, other=float("-inf"))
        gmax = tl.maximum(gmax, v)
    m = tl.max(gmax, axis=0)

    tot = tl.zeros((BLOCK,), tl.float32)
    for s in range(0, NB, BLOCK):
        idx = s + offs
        msk = idx < NB
        v = tl.load(pm_ptr + row * NB + idx, mask=msk, other=float("-inf"))
        sv = tl.load(ps_ptr + row * NB + idx, mask=msk, other=0.0)
        tot += tl.where(msk, sv * tl.exp(v - m), 0.0)
    total = tl.sum(tot, axis=0)

    x = m + tl.log(total)
    # mish(x) = x * tanh(softplus(x))
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    t = tl.exp(-2.0 * sp)
    th = (1.0 - t) / (1.0 + t)
    res = x * (x * th)
    tl.store(out_ptr + row * stride_om, res.to(out_ptr.dtype.element_ty))


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        w = self.matmul.weight
        b = self.matmul.bias
        M, K = x.shape
        N = w.shape[0]

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        NB = triton.cdiv(N, BLOCK_N)
        pm = torch.empty((M, NB), device=x.device, dtype=torch.float32)
        ps = torch.empty((M, NB), device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(M, BLOCK_M), NB)
        _gemm_lse_partial_kernel[grid](
            x, w, b, pm, ps,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            NB,
            float(self.scale_factor) * 2.0,
            float(self.clamp_min), float(self.clamp_max),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        _lse_mish_kernel[(M,)](
            pm, ps, out, NB, out.stride(0),
            BLOCK=64, num_warps=2,
        )
        return out
