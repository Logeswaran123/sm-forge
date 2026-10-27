import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt3d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    M, Cin, Cout,
    D, H, W,
    Do, Ho, Wo,
    HAS_CB: tl.constexpr,
    KS: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < Cout

    ow = offs_m % Wo
    t = offs_m // Wo
    oh = t % Ho
    t = t // Ho
    od = t % Do
    nb = t // Do

    HW = H * W
    DHW = D * HW
    KKK: tl.constexpr = KS * KS * KS

    x_batch = nb * (Cin * DHW)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kd in tl.static_range(KS):
        td = od + PAD - kd
        vd = (td >= 0) & ((td % STRIDE) == 0)
        idd = tl.where(vd, td // STRIDE, 0)
        vd = vd & (idd < D)
        for kh in tl.static_range(KS):
            th = oh + PAD - kh
            vh = (th >= 0) & ((th % STRIDE) == 0)
            ihh = tl.where(vh, th // STRIDE, 0)
            vh = vh & (ihh < H)
            for kw in tl.static_range(KS):
                tw = ow + PAD - kw
                vw = (tw >= 0) & ((tw % STRIDE) == 0)
                iww = tl.where(vw, tw // STRIDE, 0)
                vw = vw & (iww < W)

                valid = m_mask & vd & vh & vw
                base = x_batch + idd * HW + ihh * W + iww
                tap = (kd * KS + kh) * KS + kw

                for c0 in range(0, Cin, BLOCK_K):
                    ck = c0 + tl.arange(0, BLOCK_K)
                    k_mask = ck < Cin
                    a = tl.load(
                        x_ptr + base[:, None] + ck[None, :] * DHW,
                        mask=valid[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    b = tl.load(
                        w_ptr + ck[:, None] * (Cout * KKK) + offs_n[None, :] * KKK + tap,
                        mask=k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(a, b.to(a.dtype))

    if HAS_CB:
        cb = tl.load(cb_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + cb[None, :]

    # conv output rounded to output dtype (matches reference's materialized tensor)
    y = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    bb = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    res = (2.0 * y + bb[None, :]) * y + y

    S_out = Do * Ho * Wo
    sp = offs_m % S_out
    out_off = (nb * (Cout * S_out) + sp)[:, None] + offs_n[None, :] * S_out
    tl.store(out_ptr + out_off, res.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                                 padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.ks = kernel_size
        self.stride_ = stride
        self.pad = padding
        self.opad = output_padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias
        Nb, Cin, D, H, W = x.shape
        Cout = w.shape[1]
        K, S, P, OP = self.ks, self.stride_, self.pad, self.opad
        Do = (D - 1) * S - 2 * P + (K - 1) + OP + 1
        Ho = (H - 1) * S - 2 * P + (K - 1) + OP + 1
        Wo = (W - 1) * S - 2 * P + (K - 1) + OP + 1
        out = torch.empty((Nb, Cout, Do, Ho, Wo), device=x.device, dtype=x.dtype)
        M = Nb * Do * Ho * Wo
        bias = self.bias.reshape(-1).contiguous()
        has_cb = cb is not None
        cb_t = cb.contiguous() if has_cb else bias

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Cout, BLOCK_N))
        convt3d_fused_kernel[grid](
            x, w, cb_t, bias, out,
            M, Cin, Cout,
            D, H, W,
            Do, Ho, Wo,
            HAS_CB=has_cb,
            KS=K, STRIDE=S, PAD=P,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return out
