import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_pool_lse_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    Cin, D, H, W, Cout,
    PD, PH, PW, M,
    KS: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    pw = offs_m % PW
    t = offs_m // PW
    ph = t % PH
    t = t // PH
    pd = t % PD
    n = t // PD

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < Cout
    offs_k = tl.arange(0, BLOCK_K)

    HW = H * W
    DHW = D * HW
    x_base = n.to(tl.int64) * Cin * DHW
    KVOL: tl.constexpr = KS * KS * KS
    w_co_stride = Cin * KVOL

    best = tl.full((BLOCK_M, BLOCK_N), float("-inf"), dtype=tl.float32)

    for p in range(8):
        od = pd * 2 + p // 4
        oh = ph * 2 + (p // 2) % 2
        ow = pw * 2 + p % 2
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for tap in range(KVOL):
            kd = tap // (KS * KS)
            kh = (tap // KS) % KS
            kw = tap % KS
            idd = od * STRIDE - PAD + kd
            ih = oh * STRIDE - PAD + kh
            iw = ow * STRIDE - PAD + kw
            valid = mask_m & (idd >= 0) & (idd < D) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            sp = idd * HW + ih * W + iw
            for c0 in range(0, Cin, BLOCK_K):
                ci = c0 + offs_k
                mask_c = ci < Cin
                a_ptrs = x_ptr + x_base[:, None] + ci[None, :].to(tl.int64) * DHW + sp[:, None]
                a = tl.load(a_ptrs, mask=valid[:, None] & mask_c[None, :], other=0.0)
                b_ptrs = w_ptr + offs_n[None, :] * w_co_stride + ci[:, None] * KVOL + tap
                b = tl.load(b_ptrs, mask=mask_c[:, None] & mask_n[None, :], other=0.0)
                acc = tl.dot(a, b.to(a.dtype), acc)
        best = tl.maximum(best, acc)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    v = best + bias[None, :]
    v = tl.where(mask_n[None, :], v, float("-inf"))
    m = tl.max(v, axis=1)
    e = tl.exp(v - m[:, None])
    e = tl.where(mask_n[None, :], e, 0.0)
    s = tl.sum(e, axis=1)
    lse = m + tl.log(s)
    res = tl.maximum(lse, 0.0)

    out_off = ((n.to(tl.int64) * PD + pd) * PH + ph) * PW + pw
    tl.store(out_ptr + out_off, res.to(out_ptr.dtype.element_ty), mask=mask_m)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        N, Cin, D, H, W = x.shape
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        Cout = w.shape[0]
        KS, S, P = self.kernel_size, self.stride, self.padding
        Do = (D + 2 * P - KS) // S + 1
        Ho = (H + 2 * P - KS) // S + 1
        Wo = (W + 2 * P - KS) // S + 1
        PD, PH, PW = Do // 2, Ho // 2, Wo // 2
        out = torch.empty((N, 1, PD, PH, PW), device=x.device, dtype=x.dtype)
        M = N * PD * PH * PW
        if M == 0:
            return out
        BLOCK_M = 64
        BLOCK_N = max(16, triton.next_power_of_2(Cout))
        BLOCK_K = max(16, min(32, triton.next_power_of_2(Cin)))
        grid = (triton.cdiv(M, BLOCK_M),)
        conv_pool_lse_relu_kernel[grid](
            x, w, b, out,
            Cin, D, H, W, Cout,
            PD, PH, PW, M,
            KS=KS, STRIDE=S, PAD=P,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return out
