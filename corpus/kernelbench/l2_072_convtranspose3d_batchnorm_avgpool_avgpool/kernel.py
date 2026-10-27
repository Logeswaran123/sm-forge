import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fold_weights_kernel(w_ptr, b_ptr, g_ptr, beta_ptr, mean_ptr, var_ptr,
                        weff_ptr, beff_ptr, Cout, eps, inv_pool):
    # One program per (ci, co).
    pid = tl.program_id(0)
    ci = pid // Cout
    co = pid % Cout

    j = tl.arange(0, 32)          # effective tap index (27 valid)
    k = tl.arange(0, 32)          # original tap index (27 valid)
    jm = j < 27
    km = k < 27

    jd = (j // 9)[:, None]
    jh = ((j // 3) % 3)[:, None]
    jw = (j % 3)[:, None]
    kd = (k // 9)[None, :]
    kh = ((k // 3) % 3)[None, :]
    kw = (k % 3)[None, :]

    # A[j][k]: j=0 -> k in {1,2}; j=1 -> all k; j=2 -> k=0
    ad = tl.where(jd == 0, (kd >= 1).to(tl.float32),
                  tl.where(jd == 1, 1.0, (kd == 0).to(tl.float32)))
    ah = tl.where(jh == 0, (kh >= 1).to(tl.float32),
                  tl.where(jh == 1, 1.0, (kh == 0).to(tl.float32)))
    aw = tl.where(jw == 0, (kw >= 1).to(tl.float32),
                  tl.where(jw == 1, 1.0, (kw == 0).to(tl.float32)))
    A = ad * ah * aw

    wrow = tl.load(w_ptr + (ci * Cout + co) * 27 + k, mask=km, other=0.0).to(tl.float32)
    acc = tl.sum(A * wrow[None, :], axis=1)  # [32]

    g = tl.load(g_ptr + co).to(tl.float32)
    beta = tl.load(beta_ptr + co).to(tl.float32)
    mean = tl.load(mean_ptr + co).to(tl.float32)
    var = tl.load(var_ptr + co).to(tl.float32)
    s = g / tl.sqrt(var + eps)

    weff = acc * s * inv_pool
    tl.store(weff_ptr + (ci * 27 + j) * Cout + co, weff, mask=jm)

    b = tl.load(b_ptr + co).to(tl.float32)
    beff = (b - mean) * s + beta
    tl.store(beff_ptr + co, beff, mask=ci == 0)


@triton.jit
def fused_conv_kernel(x_ptr, weff_ptr, beff_ptr, out_ptr,
                      Cin, Cout, D, H, W, OD, OH, OW,
                      BLOCK: tl.constexpr, COUT_P: tl.constexpr):
    pid_s = tl.program_id(0)
    n = tl.program_id(1)

    S = OD * OH * OW
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    smask = offs < S
    ow = offs % OW
    oh = (offs // OW) % OH
    od = offs // (OW * OH)

    co = tl.arange(0, COUT_P)
    cmask = co < Cout

    acc = tl.zeros((BLOCK, COUT_P), dtype=tl.float32)
    HW = H * W
    for ci in range(Cin):
        base = x_ptr + (n * Cin + ci).to(tl.int64) * D * HW
        for j in tl.static_range(27):
            jd = j // 9
            jh = (j // 3) % 3
            jw = j % 3
            idd = 2 * od + jd
            ihh = 2 * oh + jh
            iww = 2 * ow + jw
            m = smask & (idd < D) & (ihh < H) & (iww < W)
            xv = tl.load(base + idd * HW + ihh * W + iww, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(weff_ptr + (ci * 27 + j) * Cout + co, mask=cmask, other=0.0)
            acc += xv[:, None] * wv[None, :]

    beff = tl.load(beff_ptr + co, mask=cmask, other=0.0)
    acc += beff[None, :]

    out_off = (n * Cout + co[None, :]).to(tl.int64) * S + offs[:, None]
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty),
             mask=smask[:, None] & cmask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)
        ct = self.conv_transpose
        assert tuple(ct.kernel_size) == (3, 3, 3)
        assert tuple(ct.stride) == (2, 2, 2)
        assert tuple(ct.padding) == (1, 1, 1)
        assert tuple(ct.output_padding) == (0, 0, 0)
        assert tuple(ct.dilation) == (1, 1, 1)
        assert ct.groups == 1

    def forward(self, x):
        x = x.contiguous()
        N, Cin, D, H, W = x.shape
        ct = self.conv_transpose
        bn = self.batch_norm
        Cout = ct.out_channels
        dev = x.device

        # conv-transpose output size, then two floor-mode pools of 2
        Dc, Hc, Wc = 2 * D - 1, 2 * H - 1, 2 * W - 1
        OD, OH, OW = (Dc // 2) // 2, (Hc // 2) // 2, (Wc // 2) // 2

        weff = torch.empty((Cin, 27, Cout), device=dev, dtype=torch.float32)
        beff = torch.empty((Cout,), device=dev, dtype=torch.float32)
        w = ct.weight.contiguous()
        fold_weights_kernel[(Cin * Cout,)](
            w, ct.bias.contiguous(), bn.weight.contiguous(), bn.bias.contiguous(),
            bn.running_mean.contiguous(), bn.running_var.contiguous(),
            weff, beff, Cout, float(bn.eps), 1.0 / 64.0,
        )

        out = torch.empty((N, Cout, OD, OH, OW), device=dev, dtype=x.dtype)
        S = OD * OH * OW
        if S == 0 or N == 0:
            return out
        BLOCK = 128
        COUT_P = max(triton.next_power_of_2(Cout), 2)
        grid = (triton.cdiv(S, BLOCK), N)
        fused_conv_kernel[grid](
            x, weff, beff, out,
            Cin, Cout, D, H, W, OD, OH, OW,
            BLOCK=BLOCK, COUT_P=COUT_P,
        )
        return out
