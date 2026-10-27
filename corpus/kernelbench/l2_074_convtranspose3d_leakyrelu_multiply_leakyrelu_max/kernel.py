import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_convt_leaky_mul_pool_kernel(
    x_ptr, w_ptr, b_ptr, m_ptr, out_ptr,
    C_in, C_out, D_in, H_in, W_in, Dp, Hp, Wp,
    slope,
    BLOCK: tl.constexpr, BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.program_id(1)

    S = Dp * Hp * Wp
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    smask = offs < S
    pw = offs % Wp
    t = offs // Wp
    ph = t % Hp
    pd = t // Hp

    co = tl.arange(0, BLOCK_CO)
    comask = co < C_out
    bias = tl.load(b_ptr + co, mask=comask, other=0.0).to(tl.float32)
    mult = tl.load(m_ptr + co, mask=comask, other=0.0).to(tl.float32)

    HWin = H_in * W_in
    DHWin = D_in * HWin
    x_base = x_ptr + n.to(tl.int64) * C_in * DHWin

    result = tl.full([BLOCK, BLOCK_CO], float("-inf"), tl.float32)

    for a in tl.static_range(2):
        for b in tl.static_range(2):
            for c in tl.static_range(2):
                acc = tl.zeros([BLOCK, BLOCK_CO], dtype=tl.float32)
                for ta in tl.static_range(1 + a):
                    kd = (1 - a) + 2 * ta * a
                    idd = pd + a * (1 - ta)
                    vd = idd < D_in
                    for tb in tl.static_range(1 + b):
                        kh = (1 - b) + 2 * tb * b
                        ih = ph + b * (1 - tb)
                        vh = ih < H_in
                        for tc in tl.static_range(1 + c):
                            kw = (1 - c) + 2 * tc * c
                            iw = pw + c * (1 - tc)
                            vw = iw < W_in
                            valid = smask & vd & vh & vw
                            sp = idd * HWin + ih * W_in + iw
                            tap = kd * 9 + kh * 3 + kw
                            for ci0 in range(0, C_in, BLOCK_CI):
                                ci = ci0 + tl.arange(0, BLOCK_CI)
                                cimask = ci < C_in
                                xv = tl.load(
                                    x_base + ci[None, :] * DHWin + sp[:, None],
                                    mask=valid[:, None] & cimask[None, :],
                                    other=0.0,
                                )
                                wv = tl.load(
                                    w_ptr + ci[:, None] * (C_out * 27) + co[None, :] * 27 + tap,
                                    mask=cimask[:, None] & comask[None, :],
                                    other=0.0,
                                )
                                acc = tl.dot(xv.to(tl.float16), wv.to(tl.float16), acc)
                v = acc + bias[None, :]
                v = tl.where(v >= 0, v, v * slope)
                v = v * mult[None, :]
                v = tl.where(v >= 0, v, v * slope)
                result = tl.maximum(result, v)

    out_base = out_ptr + n.to(tl.int64) * C_out * S
    optrs = out_base + co[None, :] * S + offs[:, None]
    tl.store(optrs, result.to(out_ptr.dtype.element_ty),
             mask=smask[:, None] & comask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                                 padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        conv = self.conv_transpose
        ks = conv.kernel_size
        st = conv.stride
        pd_ = conv.padding
        op = conv.output_padding
        assert tuple(ks) == (3, 3, 3) and tuple(st) == (2, 2, 2) and tuple(pd_) == (1, 1, 1)
        assert tuple(conv.dilation) == (1, 1, 1) and conv.groups == 1
        assert tuple(op) == (1, 1, 1)

        x = x.contiguous()
        N, C_in, D_in, H_in, W_in = x.shape
        C_out = conv.out_channels
        D_out = (D_in - 1) * 2 - 2 + 3 + 1
        H_out = (H_in - 1) * 2 - 2 + 3 + 1
        W_out = (W_in - 1) * 2 - 2 + 3 + 1
        Dp, Hp, Wp = D_out // 2, H_out // 2, W_out // 2

        w = conv.weight.contiguous()
        if conv.bias is not None:
            b = conv.bias.contiguous()
        else:
            b = torch.zeros(C_out, device=x.device, dtype=x.dtype)
        m = self.multiplier.reshape(-1).contiguous()

        out = torch.empty((N, C_out, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

        BLOCK = 64
        BLOCK_CI = max(16, min(32, triton.next_power_of_2(C_in)))
        BLOCK_CO = max(16, triton.next_power_of_2(C_out))
        S = Dp * Hp * Wp
        grid = (triton.cdiv(S, BLOCK), N)
        _fused_convt_leaky_mul_pool_kernel[grid](
            x, w, b, m, out,
            C_in, C_out, D_in, H_in, W_in, Dp, Hp, Wp,
            0.2,
            BLOCK=BLOCK, BLOCK_CI=BLOCK_CI, BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )
        return out
