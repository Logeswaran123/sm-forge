import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_gn_min_kernel(
    x_ptr, w_ptr, lb_ptr, gw_ptr, gb_ptr, part_ptr,
    M, N, K, eps,
    stride_xm, stride_wn, num_tiles_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr,
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
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptr + offs_n[None, :] * stride_wn + kk[:, None],
                    mask=n_mask[None, :] & k_mask[:, None], other=0.0)
        acc = tl.dot(a, w.to(a.dtype), acc)

    lb = tl.load(lb_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    y = acc + lb[None, :]
    # mimic the reference's half-precision Linear output
    y = y.to(x_ptr.dtype.element_ty).to(tl.float32)

    NG: tl.constexpr = BLOCK_N // GROUP
    y3 = tl.reshape(y, [BLOCK_M, NG, GROUP])
    mean = tl.sum(y3, axis=2) / GROUP
    d = y3 - mean[:, :, None]
    var = tl.sum(d * d, axis=2) / GROUP
    rstd = 1.0 / tl.sqrt(var + eps)
    yn3 = d * rstd[:, :, None]
    yn = tl.reshape(yn3, [BLOCK_M, BLOCK_N])

    gw = tl.load(gw_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32)
    gb = tl.load(gb_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    z = yn * gw[None, :] + gb[None, :]
    z = z.to(x_ptr.dtype.element_ty).to(tl.float32)
    z = tl.where(n_mask[None, :], z, float("inf"))

    rmin = tl.min(z, axis=1)
    tl.store(part_ptr + offs_m * num_tiles_n + pid_n, rmin, mask=m_mask)


@triton.jit
def min_bias_broadcast_kernel(
    part_ptr, bias_ptr, out_ptr,
    M, N, num_tiles_n,
    BLOCK_B: tl.constexpr, BLOCK_C: tl.constexpr, T_BLOCK: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = tl.arange(0, T_BLOCK)
    b_mask = offs_b < M
    c_mask = offs_c < N
    t_mask = offs_t < num_tiles_n

    p = tl.load(part_ptr + offs_b[:, None] * num_tiles_n + offs_t[None, :],
                mask=b_mask[:, None] & t_mask[None, :], other=float("inf"))
    rmin = tl.min(p, axis=1)  # [BLOCK_B]

    bias = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    out = bias[:, None] + rmin[None, :]  # [BLOCK_C, BLOCK_B]
    ptrs = out_ptr + offs_c[:, None] * M + offs_b[None, :]
    tl.store(ptrs, out.to(out_ptr.dtype.element_ty),
             mask=c_mask[:, None] & b_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        w = self.gemm.weight.contiguous()
        N = w.shape[0]
        G = self.group_norm.num_groups
        GROUP = N // G
        assert GROUP * G == N and (GROUP & (GROUP - 1)) == 0

        BLOCK_M, BLOCK_K = 64, 64
        BLOCK_N = max(128, GROUP)
        assert BLOCK_N % GROUP == 0
        num_tiles_n = triton.cdiv(N, BLOCK_N)

        part = torch.empty((M, num_tiles_n), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), num_tiles_n)
        gemm_gn_min_kernel[grid](
            x, w, self.gemm.bias, self.group_norm.weight, self.group_norm.bias, part,
            M, N, K, float(self.group_norm.eps),
            x.stride(0), w.stride(0), num_tiles_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP=GROUP,
            num_warps=8, num_stages=3,
        )

        out_dtype = torch.promote_types(x.dtype, self.bias.dtype)
        bias_flat = self.bias.reshape(-1)
        C = bias_flat.shape[0]
        out = torch.empty((1, C, M, 1), device=x.device, dtype=out_dtype)
        BLOCK_B, BLOCK_C = 32, 64
        T_BLOCK = triton.next_power_of_2(num_tiles_n)
        grid2 = (triton.cdiv(C, BLOCK_C), triton.cdiv(M, BLOCK_B))
        min_bias_broadcast_kernel[grid2](
            part, bias_flat, out, M, C, num_tiles_n,
            BLOCK_B=BLOCK_B, BLOCK_C=BLOCK_C, T_BLOCK=T_BLOCK, num_warps=4,
        )
        return out
