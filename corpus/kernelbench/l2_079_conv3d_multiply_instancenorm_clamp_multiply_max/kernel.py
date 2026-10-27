import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_mul_kernel(
    x_ptr, w_ptr, b_ptr, m_ptr, y_ptr,
    C_out, D, H, W, OH, OW, S,
    C_IN: tl.constexpr, K: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.program_id(1).to(tl.int64)

    offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = offs < S
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    od = t // OH

    cofs = tl.arange(0, BLOCK_C)
    cmask = cofs < C_out

    HW = H * W
    DHW = D * HW
    x_base = x_ptr + n * C_IN * DHW
    sp_base = od * HW + oh * W + ow

    acc = tl.zeros([BLOCK_S, BLOCK_C], dtype=tl.float32)
    KKK: tl.constexpr = K * K * K
    for ci in tl.static_range(C_IN):
        for kd in tl.static_range(K):
            for kh in tl.static_range(K):
                for kw in tl.static_range(K):
                    xv = tl.load(
                        x_base + ci * DHW + sp_base + (kd * HW + kh * W + kw),
                        mask=smask, other=0.0,
                    ).to(tl.float32)
                    wv = tl.load(
                        w_ptr + cofs * (C_IN * KKK) + (ci * KKK + kd * K * K + kh * K + kw),
                        mask=cmask, other=0.0,
                    ).to(tl.float32)
                    acc += xv[:, None] * wv[None, :]

    bias = tl.load(b_ptr + cofs, mask=cmask, other=0.0).to(tl.float32)
    mult = tl.load(m_ptr + cofs, mask=cmask, other=0.0).to(tl.float32)
    acc = (acc + bias[None, :]) * mult[None, :]

    y_ptrs = y_ptr + (n * C_out + cofs[None, :]) * S + offs[:, None]
    tl.store(y_ptrs, acc, mask=smask[:, None] & cmask[None, :])


@triton.jit
def instnorm_stats_kernel(
    y_ptr, mean_ptr, rstd_ptr, S, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    base = y_ptr + pid * S

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, S, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        v = tl.load(base + offs, mask=offs < S, other=0.0)
        acc += v
    mean = tl.sum(acc, axis=0) / S

    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, S, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        msk = offs < S
        v = tl.load(base + offs, mask=msk, other=0.0)
        d = tl.where(msk, v - mean, 0.0)
        acc2 += d * d
    var = tl.sum(acc2, axis=0) / S
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def norm_clamp_mul_max_kernel(
    y_ptr, mean_ptr, rstd_ptr, m_ptr, out_ptr,
    C_out, S, clamp_min, clamp_max,
    BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.program_id(1).to(tl.int64)

    offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = offs < S
    cofs = tl.arange(0, BLOCK_C)
    cmask = cofs < C_out

    mean = tl.load(mean_ptr + n * C_out + cofs, mask=cmask, other=0.0)
    rstd = tl.load(rstd_ptr + n * C_out + cofs, mask=cmask, other=0.0)
    mult = tl.load(m_ptr + cofs, mask=cmask, other=0.0).to(tl.float32)

    y_ptrs = y_ptr + (n * C_out + cofs[:, None]) * S + offs[None, :]
    m2 = cmask[:, None] & smask[None, :]
    v = tl.load(y_ptrs, mask=m2, other=0.0)

    v = (v - mean[:, None]) * rstd[:, None]
    v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
    v = v * mult[:, None]
    v = tl.where(cmask[:, None], v, float("-inf"))
    r = tl.max(v, axis=0)

    tl.store(out_ptr + n * S + offs, r.to(out_ptr.dtype.element_ty), mask=smask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = x.contiguous()
        N, C_in, D, H, W = x.shape
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        C_out = w.shape[0]
        K = w.shape[2]
        OD, OH, OW = D - K + 1, H - K + 1, W - K + 1
        S = OD * OH * OW
        mult = self.multiplier.contiguous().view(-1)

        y = torch.empty((N, C_out, S), device=x.device, dtype=torch.float32)
        BLOCK_C = triton.next_power_of_2(C_out)
        BLOCK_S = 128
        grid = (triton.cdiv(S, BLOCK_S), N)
        conv3d_mul_kernel[grid](
            x, w, b, mult, y,
            C_out, D, H, W, OH, OW, S,
            C_IN=C_in, K=K, BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        mean = torch.empty((N * C_out,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N * C_out,), device=x.device, dtype=torch.float32)
        instnorm_stats_kernel[(N * C_out,)](
            y, mean, rstd, S, float(self.instance_norm.eps),
            BLOCK=1024, num_warps=4,
        )

        out = torch.empty((N, OD, OH, OW), device=x.device, dtype=x.dtype)
        BLOCK_S2 = 256
        grid2 = (triton.cdiv(S, BLOCK_S2), N)
        norm_clamp_mul_max_kernel[grid2](
            y, mean, rstd, mult, out,
            C_out, S, float(self.clamp_min), float(self.clamp_max),
            BLOCK_S=BLOCK_S2, BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out
