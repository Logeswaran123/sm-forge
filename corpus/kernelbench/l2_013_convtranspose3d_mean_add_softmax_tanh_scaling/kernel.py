import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _depth_sum_kernel(
    x_ptr, s_ptr,
    CIN, D, HW, D_OUT, PAD,
    K: tl.constexpr, KPAD: tl.constexpr, BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_bc = tl.program_id(1)
    b = pid_bc // CIN
    ci = pid_bc % CIN

    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    smask = offs < HW
    ks = tl.arange(0, KPAD)
    lo = tl.maximum(PAD - ks, 0)
    hi = tl.minimum(D - 1, D_OUT - 1 + PAD - ks)
    kvalid = ks < K

    acc = tl.zeros((KPAD, BLOCK), dtype=tl.float32)
    base = (b * CIN + ci).to(tl.int64) * D * HW
    for d in range(0, D):
        v = tl.load(x_ptr + base + d * HW + offs, mask=smask, other=0.0).to(tl.float32)
        sel = (d >= lo) & (d <= hi) & kvalid
        acc += tl.where(sel[:, None], v[None, :], 0.0)

    # S layout: [B, K, CIN, HW]
    out_off = ((b * K + ks[:, None]).to(tl.int64) * CIN + ci) * HW + offs[None, :]
    tl.store(s_ptr + out_off, acc, mask=kvalid[:, None] & smask[None, :])


@triton.jit
def _conv2d_softmax_kernel(
    s_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    CIN, COUT, H, W, HO, WO, PAD,
    inv_dout, scale,
    HAS_CB: tl.constexpr,
    K: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    pid_p = tl.program_id(0)
    b = tl.program_id(1)

    HW = H * W
    HOWO = HO * WO
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    pmask = offs_p < HOWO
    oh = offs_p // WO
    ow = offs_p % WO

    co = tl.arange(0, BLOCK_CO)
    comask = co < COUT

    K3: tl.constexpr = K * K * K
    acc = tl.zeros((BLOCK_P, BLOCK_CO), dtype=tl.float32)

    for kd in range(0, K):
        for ci in range(0, CIN):
            s_base = ((b * K + kd).to(tl.int64) * CIN + ci) * HW
            w_base = ci * COUT * K3 + kd * K * K
            for kh in tl.static_range(K):
                ih = oh + PAD - kh
                hvalid = (ih >= 0) & (ih < H)
                for kw in tl.static_range(K):
                    iw = ow + PAD - kw
                    valid = pmask & hvalid & (iw >= 0) & (iw < W)
                    sv = tl.load(s_ptr + s_base + ih * W + iw, mask=valid, other=0.0)
                    wv = tl.load(w_ptr + w_base + co * K3 + kh * K + kw,
                                 mask=comask, other=0.0).to(tl.float32)
                    acc += sv[:, None] * wv[None, :]

    z = acc * inv_dout
    if HAS_CB:
        cb = tl.load(cb_ptr + co, mask=comask, other=0.0).to(tl.float32)
        z += cb[None, :]
    bb = tl.load(b_ptr + co, mask=comask, other=0.0).to(tl.float32)
    z += bb[None, :]

    z = tl.where(comask[None, :], z, float("-inf"))
    m = tl.max(z, axis=1)
    e = tl.exp(z - m[:, None])
    e = tl.where(comask[None, :], e, 0.0)
    ssum = tl.sum(e, axis=1)
    sm = e / ssum[:, None]
    # tanh(x) = 1 - 2/(exp(2x)+1), x in [0,1] so stable
    t = 1.0 - 2.0 / (tl.exp(2.0 * sm) + 1.0)
    y = t * scale

    out_off = (b * COUT + co[None, :]).to(tl.int64) * HOWO + offs_p[:, None]
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty),
             mask=pmask[:, None] & comask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        conv = self.conv_transpose
        K = conv.kernel_size[0]
        assert conv.kernel_size == (K, K, K)
        assert conv.stride == (1, 1, 1)
        P = conv.padding[0]
        assert conv.padding == (P, P, P)

        x = x.contiguous()
        B, CIN, D, H, W = x.shape
        COUT = conv.out_channels
        D_OUT = D - 1 - 2 * P + K
        HO = H - 1 - 2 * P + K
        WO = W - 1 - 2 * P + K
        HW = H * W

        weight = conv.weight.contiguous()  # [CIN, COUT, K, K, K]
        cbias = conv.bias
        has_cb = cbias is not None
        cb = cbias.contiguous() if has_cb else weight
        bias = self.bias.reshape(-1).contiguous()

        KPAD = triton.next_power_of_2(K)
        S = torch.empty((B, K, CIN, H, W), device=x.device, dtype=torch.float32)
        BLOCK = 256
        grid1 = (triton.cdiv(HW, BLOCK), B * CIN)
        _depth_sum_kernel[grid1](
            x, S, CIN, D, HW, D_OUT, P,
            K=K, KPAD=KPAD, BLOCK=BLOCK, num_warps=4,
        )

        out = torch.empty((B, COUT, 1, HO, WO), device=x.device, dtype=x.dtype)
        BLOCK_P = 64
        BLOCK_CO = triton.next_power_of_2(COUT)
        grid2 = (triton.cdiv(HO * WO, BLOCK_P), B)
        _conv2d_softmax_kernel[grid2](
            S, weight, cb, bias, out,
            CIN, COUT, H, W, HO, WO, P,
            1.0 / D_OUT, float(self.scaling_factor),
            HAS_CB=has_cb, K=K, BLOCK_P=BLOCK_P, BLOCK_CO=BLOCK_CO, num_warps=4,
        )
        return out
