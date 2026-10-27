import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_sub_hswish_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    NB, C, H, W, OC, PH, PW, K,
    sub_val,
    KS: tl.constexpr, P: tl.constexpr, HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = NB * PH * PW
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    n_idx = offs_m // (PH * PW)
    r = offs_m % (PH * PW)
    ph = r // PW
    pw = r % PW

    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < OC

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BN], dtype=tl.float32)

    best = tl.full([BM, BN], float("-inf"), dtype=tl.float32)

    for dy in tl.static_range(P):
        for dx in tl.static_range(P):
            oh = ph * P + dy
            ow = pw * P + dx
            base = n_idx * (C * H * W) + oh * W + ow
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            for k0 in range(0, K, BK):
                offs_k = k0 + tl.arange(0, BK)
                k_mask = offs_k < K
                c = offs_k // (KS * KS)
                rr = offs_k % (KS * KS)
                kh = rr // KS
                kw = rr % KS
                koff = c * (H * W) + kh * W + kw
                a = tl.load(
                    x_ptr + base[:, None] + koff[None, :],
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    w_ptr + offs_n[None, :] * K + offs_k[:, None],
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
                acc += tl.dot(a, b.to(a.dtype))
            v = acc + bias[None, :] - sub_val
            hs = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0)
            v = v * hs / 6.0
            best = tl.maximum(best, v)

    # Mish: x * tanh(softplus(x))
    xv = best
    sp = tl.where(xv > 20.0, xv, tl.log(1.0 + tl.exp(tl.minimum(xv, 20.0))))
    e = tl.exp(-2.0 * sp)  # sp >= 0, so stable
    t = (1.0 - e) / (1.0 + e)
    y = xv * t

    out_off = (n_idx[:, None] * OC + offs_n[None, :]) * (PH * PW) + (ph * PW + pw)[:, None]
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        NB, C, H, W = x.shape
        w = self.conv.weight.contiguous()
        OC = w.shape[0]
        KS = w.shape[2]
        OH = H - KS + 1
        OW = W - KS + 1
        P = self.pool_kernel_size
        if isinstance(P, (tuple, list)):
            P = P[0]
        PH = OH // P
        PW = OW // P
        K = C * KS * KS

        out = torch.empty((NB, OC, PH, PW), device=x.device, dtype=x.dtype)
        has_bias = self.conv.bias is not None
        b = self.conv.bias.contiguous() if has_bias else w

        BM, BN, BK = 64, 64, 32
        M = NB * PH * PW
        grid = (triton.cdiv(M, BM), triton.cdiv(OC, BN))
        conv_sub_hswish_pool_mish_kernel[grid](
            x, w, b, out,
            NB, C, H, W, OC, PH, PW, K,
            float(self.subtract_value),
            KS=KS, P=P, HAS_BIAS=has_bias,
            BM=BM, BN=BN, BK=BK,
            num_warps=4,
        )
        return out
