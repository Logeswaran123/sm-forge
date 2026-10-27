import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_scale_bn_kernel(
    x_ptr, w_ptr, b_ptr, s_ptr, rm_ptr, rv_ptr, g_ptr, beta_ptr, out_ptr,
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
        # W is [N, K]; load a [BLOCK_K, BLOCK_N] tile of W^T
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
            mask=mask_n[None, :] & mask_k[:, None], other=0.0,
        )
        b = b.to(a.dtype)
        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    scale = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    rmean = tl.load(rm_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    rvar = tl.load(rv_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    gamma = tl.load(g_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    inv = gamma / tl.sqrt(rvar + eps)
    y = (acc + bias[None, :]) * scale[None, :]
    y = (y - rmean[None, :]) * inv[None, :] + beta[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        w = self.gemm.weight
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        gemm_scale_bn_kernel[grid](
            x, w, self.gemm.bias, self.scale,
            self.bn.running_mean, self.bn.running_var,
            self.bn.weight, self.bn.bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            float(self.bn.eps),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )
        return out
