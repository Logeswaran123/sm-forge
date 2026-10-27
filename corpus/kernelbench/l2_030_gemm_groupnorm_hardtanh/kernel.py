import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
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
        acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _groupnorm_hardtanh_kernel(
    y_ptr, gamma_ptr, beta_ptr, out_ptr,
    N, C_PER_G,
    stride_ym, stride_om,
    eps, hmin, hmax,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.program_id(1)

    offs = tl.arange(0, BLOCK_C)
    ch = g * C_PER_G + offs
    mask = (offs < C_PER_G) & (ch < N)

    v = tl.load(y_ptr + row * stride_ym + ch, mask=mask, other=0.0).to(tl.float32)
    cnt = C_PER_G.to(tl.float32)
    mean = tl.sum(v, axis=0) / cnt
    d = tl.where(mask, v - mean, 0.0)
    var = tl.sum(d * d, axis=0) / cnt
    rstd = 1.0 / tl.sqrt(var + eps)

    gam = tl.load(gamma_ptr + ch, mask=mask, other=1.0).to(tl.float32)
    bet = tl.load(beta_ptr + ch, mask=mask, other=0.0).to(tl.float32)
    o = d * rstd * gam + bet
    o = tl.minimum(tl.maximum(o, hmin), hmax)

    tl.store(out_ptr + row * stride_om + ch, o.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        M, K = x2.shape
        w = self.gemm.weight
        b = self.gemm.bias
        N = w.shape[0]

        y = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BM, BN, BK = 64, 128, 64
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _linear_kernel[grid](
            x2, w, b, y,
            M, N, K,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=4, num_stages=3,
        )

        G = self.group_norm.num_groups
        C_PER_G = N // G
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        BLOCK_C = triton.next_power_of_2(C_PER_G)
        _groupnorm_hardtanh_kernel[(M, G)](
            y, self.group_norm.weight, self.group_norm.bias, out,
            N, C_PER_G,
            y.stride(0), out.stride(0),
            float(self.group_norm.eps),
            float(self.hardtanh.min_val), float(self.hardtanh.max_val),
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out.reshape(*orig_shape[:-1], N)
