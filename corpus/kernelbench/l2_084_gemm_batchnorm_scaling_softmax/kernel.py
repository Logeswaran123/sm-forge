import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_bn_scale_kernel(
    x_ptr, w_ptr, bias_ptr, rmean_ptr, rvar_ptr, gamma_ptr, beta_ptr, scale_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        mask_k = kk < K
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        )
        acc = tl.dot(a, b.to(a.dtype), acc)

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    rmean = tl.load(rmean_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    rvar = tl.load(rvar_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(rvar + eps)
    y = acc + bias[None, :]
    y = (y - rmean[None, :]) * (inv_std * gamma)[None, :] + beta[None, :]
    y = y * scale

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        y, mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def softmax_kernel(
    in_ptr, out_ptr, N,
    stride_im, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    in_row = in_ptr + row * stride_im
    out_row = out_ptr + row * stride_om

    m_vec = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    s_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK):
        idx = n0 + offs
        mask = idx < N
        v = tl.load(in_row + idx, mask=mask, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m_vec, v)
        # guard against (-inf) - (-inf)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        s_vec = s_vec * tl.exp(m_vec - m_safe) + tl.exp(v - m_safe)
        m_vec = m_new

    m = tl.max(m_vec, axis=0)
    s = tl.sum(s_vec * tl.exp(m_vec - m), axis=0)
    inv_s = 1.0 / s

    for n0 in range(0, N, BLOCK):
        idx = n0 + offs
        mask = idx < N
        v = tl.load(in_row + idx, mask=mask, other=float("-inf")).to(tl.float32)
        p = tl.exp(v - m) * inv_s
        tl.store(out_row + idx, p.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        w = self.gemm.weight
        N = w.shape[0]

        tmp = torch.empty((M, N), device=x.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        gemm_bn_scale_kernel[grid](
            x, w, self.gemm.bias, self.bn.running_mean, self.bn.running_var,
            self.bn.weight, self.bn.bias, self.scale,
            tmp,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            tmp.stride(0), tmp.stride(1),
            float(self.bn.eps),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        softmax_kernel[(M,)](
            tmp, out, N,
            tmp.stride(0), out.stride(0),
            BLOCK=1024, num_warps=4,
        )
        return out
