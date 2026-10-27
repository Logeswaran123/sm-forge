import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
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
        # W is [N, K]; load a [BLOCK_K, BLOCK_N] tile of W^T
        b = tl.load(
            w_ptr + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=k_mask[:, None] & n_mask[None, :], other=0.0,
        )
        acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def gn_swish_kernel(
    y_ptr, gamma_ptr, beta_ptr, mw_ptr, out_ptr,
    C, GROUP_SIZE, eps,
    stride_ym, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP_SIZE
    cols = g * GROUP_SIZE + offs

    v = tl.load(y_ptr + row * stride_ym + cols, mask=mask, other=0.0).to(tl.float32)
    n = GROUP_SIZE.to(tl.float32)
    mean = tl.sum(v, axis=0) / n
    d = tl.where(mask, v - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(mw_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    h = d * rstd * gamma + beta
    h = h * tl.sigmoid(h)
    h = h * mw
    h = h * tl.sigmoid(h)

    tl.store(out_ptr + row * stride_om + cols, h.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        w = self.gemm.weight
        b = self.gemm.bias
        N = w.shape[0]

        y = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_kernel[grid](
            x, w, b, y,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        G = self.group_norm.num_groups
        group_size = N // G
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        BLOCK = max(triton.next_power_of_2(group_size), 16)
        gn_swish_kernel[(M, G)](
            y, self.group_norm.weight, self.group_norm.bias, self.multiply_weight, out,
            N, group_size, float(self.group_norm.eps),
            y.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=1,
        )
        return out
