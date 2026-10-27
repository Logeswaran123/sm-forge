import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, z_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_zm, stride_zn,
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

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    tl.store(
        z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def norm_res_kernel(
    z_ptr, y_ptr, out_ptr,
    N, eps,
    stride_zm, stride_ym, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    z_row = z_ptr + row * stride_zm
    y_row = y_ptr + row * stride_ym
    o_row = out_ptr + row * stride_om

    # mean
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK):
        idx = n0 + offs
        v = tl.load(z_row + idx, mask=idx < N, other=0.0)
        s += v
    mean = tl.sum(s, axis=0) / N

    # biased variance
    s2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK):
        idx = n0 + offs
        m = idx < N
        v = tl.load(z_row + idx, mask=m, other=0.0)
        d = tl.where(m, v - mean, 0.0)
        s2 += d * d
    var = tl.sum(s2, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for n0 in range(0, N, BLOCK):
        idx = n0 + offs
        m = idx < N
        v = tl.load(z_row + idx, mask=m, other=0.0)
        yv = tl.load(y_row + idx, mask=m, other=0.0).to(tl.float32)
        r = ((v - mean) * rstd + yv) * yv
        tl.store(o_row + idx, r.to(out_ptr.dtype.element_ty), mask=m)


class Model(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        w = self.bmm.weight
        b = self.bmm.bias
        M, K = x.shape
        N = w.shape[0]

        z = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BM, BN, BK = 128, 128, 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        linear_kernel[grid](
            x, w, b, z,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            z.stride(0), z.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=8, num_stages=3,
        )

        out_dtype = torch.promote_types(x.dtype, y.dtype)
        out = torch.empty((M, N), device=x.device, dtype=out_dtype)
        norm_res_kernel[(M,)](
            z, y, out,
            N, float(self.instance_norm.eps),
            z.stride(0), y.stride(0), out.stride(0),
            BLOCK=1024, num_warps=4,
        )
        return out
