import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_swish_bias_gn_kernel(
    x_ptr, w_ptr, lb_ptr, b_ptr, gamma_ptr, beta_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    eps,
    CPG: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, BLOCK_N)
    col_mask = offs_c < CPG
    offs_n = pid_g * CPG + offs_c
    n_mask = col_mask & (offs_n < N)
    m_mask = offs_m < M
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None], other=0.0,
        )
        acc = tl.dot(a, w.to(a.dtype), acc)

    lb = tl.load(lb_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    y = acc + lb[None, :]
    y = y * tl.sigmoid(y)
    y = y + bb[None, :]

    cm = n_mask[None, :]
    y_masked = tl.where(cm, y, 0.0)
    mean = tl.sum(y_masked, axis=1) / CPG
    d = tl.where(cm, y - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / CPG
    rstd = 1.0 / tl.sqrt(var + eps)
    out = d * rstd[:, None] * gamma[None, :] + beta[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        w = self.matmul.weight
        N = w.shape[0]
        G = self.group_norm.num_groups
        CPG = N // G
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M = 128
        BLOCK_N = triton.next_power_of_2(max(CPG, 16))
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), G)
        fused_linear_swish_bias_gn_kernel[grid](
            x, w, self.matmul.bias, self.bias,
            self.group_norm.weight, self.group_norm.bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            float(self.group_norm.eps),
            CPG=CPG,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return out
