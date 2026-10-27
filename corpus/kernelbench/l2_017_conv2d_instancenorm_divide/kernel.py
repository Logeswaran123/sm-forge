import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    C, H, W, OC, OH, OW,
    KS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n = tl.program_id(2)

    P = OH * OW
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < P
    n_mask = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    Ktot = C * KS * KS
    x_base = x_ptr + n.to(tl.int64) * C * H * W

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, Ktot, BK):
        offs_k = k0 + tl.arange(0, BK)
        k_mask = offs_k < Ktot
        ci = offs_k // (KS * KS)
        r = offs_k % (KS * KS)
        kh = r // KS
        kw = r % KS

        a_ptrs = (x_base + ci[None, :] * (H * W)
                  + (oh[:, None] + kh[None, :]) * W
                  + (ow[:, None] + kw[None, :]))
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float16)

        b_ptrs = w_ptr + offs_n[None, :] * Ktot + offs_k[:, None]
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)

        acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    y_ptrs = (y_ptr + (n.to(tl.int64) * OC + offs_n[None, :].to(tl.int64)) * P
              + offs_m[:, None])
    tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def instnorm_div_kernel(
    y_ptr, out_ptr, P, inv_div, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * P
    offs = tl.arange(0, BLOCK)

    # pass 1: mean
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, P, BLOCK):
        idx = start + offs
        msk = idx < P
        v = tl.load(y_ptr + base + idx, mask=msk, other=0.0)
        s += v
    mean = tl.sum(s, axis=0) / P

    # pass 2: variance (biased)
    s2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, P, BLOCK):
        idx = start + offs
        msk = idx < P
        v = tl.load(y_ptr + base + idx, mask=msk, other=0.0)
        d = tl.where(msk, v - mean, 0.0)
        s2 += d * d
    var = tl.sum(s2, axis=0) / P
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = rstd * inv_div

    # pass 3: normalize and divide
    for start in range(0, P, BLOCK):
        idx = start + offs
        msk = idx < P
        v = tl.load(y_ptr + base + idx, mask=msk, other=0.0)
        o = (v - mean) * scale
        tl.store(out_ptr + base + idx, o.to(out_ptr.dtype.element_ty), mask=msk)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        OC = w.shape[0]
        KS = w.shape[2]
        OH = H - KS + 1
        OW = W - KS + 1
        P = OH * OW

        y = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)
        BM, BN, BK = 64, 64, 32
        grid = (triton.cdiv(P, BM), triton.cdiv(OC, BN), N)
        conv2d_implicit_gemm_kernel[grid](
            x, w, b, y, C, H, W, OC, OH, OW,
            KS=KS, BM=BM, BN=BN, BK=BK, num_warps=4,
        )

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
        eps = float(self.instance_norm.eps)
        instnorm_div_kernel[(N * OC,)](
            y, out, P, 1.0 / float(self.divide_by), eps,
            BLOCK=1024, num_warps=4,
        )
        return out
