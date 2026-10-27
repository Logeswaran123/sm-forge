import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_scale_bn_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr,
    rm_ptr, rv_ptr, gamma_ptr, beta_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    eps,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        mask_k = kk < K
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        # W is [N, K]; load a [BK, BN] tile (transposed view)
        w_ptrs = w_ptr + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        w = w.to(a.dtype)
        acc = tl.dot(a, w, acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    rm = tl.load(rm_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    rv = tl.load(rv_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    v = (acc + bias[None, :]) * scale[None, :]
    inv = 1.0 / tl.sqrt(rv + eps)
    y = (v - rm[None, :]) * (inv * gamma)[None, :] + beta[None, :]

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, y.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        K = x.shape[-1]
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        w = self.gemm.weight
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BM, BN, BK = 64, 128, 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        gemm_scale_bn_kernel[grid](
            x2, w, self.gemm.bias, self.scale,
            self.bn.running_mean, self.bn.running_var,
            self.bn.weight, self.bn.bias,
            out,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            float(self.bn.eps),
            BM=BM, BN=BN, BK=BK,
            num_warps=4, num_stages=3,
        )
        return out.reshape(*x.shape[:-1], N)
