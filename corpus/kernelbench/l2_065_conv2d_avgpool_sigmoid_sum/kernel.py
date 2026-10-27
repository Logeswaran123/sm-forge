import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_partial_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    sxb, sxc, sxh, sxw,
    PH, PW, NBLK,
    CIN: tl.constexpr, COUT: tl.constexpr, COUT_P: tl.constexpr,
    KS: tl.constexpr, POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    bidx = tl.program_id(1)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    npos = PH * PW
    pmask = offs_p < npos
    ph = offs_p // PW
    pw = offs_p % PW

    offs_c = tl.arange(0, COUT_P)
    cmask = offs_c < COUT

    acc = tl.zeros((BLOCK_P, COUT_P), dtype=tl.float32)

    x_base = x_ptr + bidx.to(tl.int64) * sxb
    row0 = ph * POOL
    col0 = pw * POOL

    for ci in range(CIN):
        xc = x_base + ci * sxc
        for kh in tl.static_range(KS):
            for kw in tl.static_range(KS):
                s = tl.zeros((BLOCK_P,), dtype=tl.float32)
                for i in tl.static_range(POOL):
                    for j in tl.static_range(POOL):
                        r = row0 + (i + kh)
                        c = col0 + (j + kw)
                        v = tl.load(xc + r * sxh + c * sxw, mask=pmask, other=0.0)
                        s += v.to(tl.float32)
                wv = tl.load(
                    w_ptr + offs_c * (CIN * KS * KS) + ci * (KS * KS) + kh * KS + kw,
                    mask=cmask, other=0.0,
                ).to(tl.float32)
                acc += s[:, None] * wv[None, :]

    bias = tl.load(b_ptr + offs_c, mask=cmask, other=0.0).to(tl.float32)
    pooled = acc * (1.0 / (POOL * POOL)) + bias[None, :]
    sig = 1.0 / (1.0 + tl.exp(-pooled))
    valid = pmask[:, None] & cmask[None, :]
    sig = tl.where(valid, sig, 0.0)
    total = tl.sum(tl.sum(sig, axis=1), axis=0)
    tl.store(part_ptr + bidx * NBLK + pid_p, total)


@triton.jit
def reduce_partials_kernel(part_ptr, out_ptr, NBLK, BLOCK: tl.constexpr):
    bidx = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, NBLK, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < NBLK
        acc += tl.load(part_ptr + bidx * NBLK + offs, mask=m, other=0.0)
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + bidx, total.to(out_ptr.dtype.element_ty))


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool = pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        B, C, H, W = x.shape
        KS = self.kernel_size
        P = self.pool
        OH = H - KS + 1
        OW = W - KS + 1
        PH = OH // P
        PW = OW // P
        npos = PH * PW

        w = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        BLOCK_P = 64
        NBLK = triton.cdiv(npos, BLOCK_P)
        COUT = self.out_channels
        COUT_P = triton.next_power_of_2(COUT)

        part = torch.empty((B, NBLK), device=x.device, dtype=torch.float32)
        out = torch.empty((B,), device=x.device, dtype=x.dtype)

        conv_pool_sigmoid_partial_kernel[(NBLK, B)](
            x, w, bias, part,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            PH, PW, NBLK,
            CIN=C, COUT=COUT, COUT_P=COUT_P,
            KS=KS, POOL=P,
            BLOCK_P=BLOCK_P,
            num_warps=4,
        )
        reduce_partials_kernel[(B,)](part, out, NBLK, BLOCK=256, num_warps=4)
        return out
