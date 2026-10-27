import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, sc_ptr, bi_ptr, out_ptr,
    Cin, Cout, D, H, W, OD, OH, OW,
    K: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    pid_s = tl.program_id(0)
    n = tl.program_id(1).to(tl.int64)

    OSP = OD * OH * OW
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = offs_s < OSP
    ow = offs_s % OW
    t = offs_s // OW
    oh = t % OH
    od = t // OH

    offs_co = tl.arange(0, BLOCK_CO)
    co_mask = offs_co < Cout

    acc = tl.zeros((BLOCK_S, BLOCK_CO), dtype=tl.float32)

    HW = H * W
    DHW = D * HW
    x_base = x_ptr + n * Cin * DHW
    KKK: tl.constexpr = K * K * K

    for c in range(Cin):
        x_c = x_base + c * DHW
        w_c = w_ptr + offs_co * (Cin * KKK) + c * KKK
        for kd in tl.static_range(K):
            for kh in tl.static_range(K):
                for kw in tl.static_range(K):
                    xoff = (od + kd) * HW + (oh + kh) * W + (ow + kw)
                    xv = tl.load(x_c + xoff, mask=s_mask, other=0.0).to(tl.float32)
                    wv = tl.load(w_c + (kd * K * K + kh * K + kw),
                                 mask=co_mask, other=0.0).to(tl.float32)
                    acc += xv[:, None] * wv[None, :]

    cb = tl.load(cb_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    sc = tl.load(sc_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    bi = tl.load(bi_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)

    z = (acc + cb[None, :]) * sc[None, :]
    # tanh(z) = 1 - 2 / (exp(2z) + 1)
    th = 1.0 - 2.0 / (tl.exp(2.0 * z) + 1.0)
    u = th * bi[None, :]
    y = 1.0 / (1.0 + tl.exp(-u))

    out_base = out_ptr + n * Cout * OSP
    optrs = out_base + offs_co[None, :] * OSP + offs_s[:, None]
    omask = s_mask[:, None] & co_mask[None, :]
    tl.store(optrs, y.to(out_ptr.dtype.element_ty), mask=omask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        N, Cin, D, H, W = x.shape
        w = self.conv.weight.contiguous()
        Cout = w.shape[0]
        K = w.shape[2]
        OD, OH, OW = D - K + 1, H - K + 1, W - K + 1
        cb = self.conv.bias.contiguous()
        sc = self.scaling_factor.contiguous().view(-1)
        bi = self.bias.contiguous().view(-1)

        out = torch.empty((N, Cout, OD, OH, OW), device=x.device, dtype=x.dtype)
        OSP = OD * OH * OW
        BLOCK_S = 128
        BLOCK_CO = max(triton.next_power_of_2(Cout), 2)
        grid = (triton.cdiv(OSP, BLOCK_S), N)
        conv3d_fused_kernel[grid](
            x, w, cb, sc, bi, out,
            Cin, Cout, D, H, W, OD, OH, OW,
            K=K, BLOCK_S=BLOCK_S, BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )
        return out
