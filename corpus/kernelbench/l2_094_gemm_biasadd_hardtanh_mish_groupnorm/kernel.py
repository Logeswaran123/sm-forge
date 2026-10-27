import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_bias_act_kernel(
    x_ptr, w_ptr, lb_ptr, b_ptr, y_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0,
        )
        b = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (kk[:, None] < K), other=0.0,
        )
        acc += tl.dot(a, b.to(a.dtype))

    nmask = offs_n < N
    lb = tl.load(lb_ptr + offs_n, mask=nmask, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + offs_n, mask=nmask, other=0.0).to(tl.float32)
    v = acc + lb[None, :] + bb[None, :]
    # hardtanh
    v = tl.minimum(tl.maximum(v, -1.0), 1.0)
    # mish: v * tanh(softplus(v)); v in [-1, 1] so this is stable
    sp = tl.log(1.0 + tl.exp(v))
    e = tl.exp(-2.0 * sp)
    t = (1.0 - e) / (1.0 + e)
    v = v * t

    mask = (offs_m[:, None] < M) & nmask[None, :]
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, v, mask=mask)


@triton.jit
def groupnorm_kernel(
    y_ptr, w_ptr, b_ptr, out_ptr,
    N, G, CPG, eps,
    stride_ym, stride_om,
    BLOCK_G: tl.constexpr, BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    offs_c = tl.arange(0, BLOCK_C)
    gmask = offs_g < G
    cmask = offs_c < CPG
    mask = gmask[:, None] & cmask[None, :]
    ch = offs_g[:, None] * CPG + offs_c[None, :]

    v = tl.load(y_ptr + row * stride_ym + ch, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(v, axis=1) / CPG
    d = tl.where(mask, v - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / CPG
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + ch, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + ch, mask=mask, other=0.0).to(tl.float32)
    o = d * rstd[:, None] * w + b
    tl.store(out_ptr + row * stride_om + ch, o.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        M, K = x2.shape
        W = self.gemm.weight
        N = W.shape[0]

        y = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BM, BN, BK = 64, 128, 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        gemm_bias_act_kernel[grid](
            x2, W, self.gemm.bias, self.bias, y,
            M, N, K,
            x2.stride(0), x2.stride(1),
            W.stride(0), W.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=4, num_stages=3,
        )

        G = self.groupnorm.num_groups
        CPG = N // G
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        BLOCK_C = triton.next_power_of_2(CPG)
        BLOCK_G = max(1, min(triton.next_power_of_2(G), 4096 // BLOCK_C))
        grid2 = (M, triton.cdiv(G, BLOCK_G))
        groupnorm_kernel[grid2](
            y, self.groupnorm.weight, self.groupnorm.bias, out,
            N, G, CPG, float(self.groupnorm.eps),
            y.stride(0), out.stride(0),
            BLOCK_G=BLOCK_G, BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out.reshape(*orig_shape[:-1], N)
