import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt3d_ln_gelu_kernel(
    x_ptr, w_ptr, b_ptr, lnw_ptr, lnb_ptr, out_ptr,
    IC, OC, D, H, W, D_out, H_out, W_out,
    eps, scale,
    S: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_oc = tl.program_id(1)

    oh = pid_row % H_out
    tmp = pid_row // H_out
    od = tmp % D_out
    b = tmp // D_out

    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ow = tl.arange(0, BLOCK_W)
    oc_mask = oc < OC
    ow_mask = ow < W_out

    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

    KKK: tl.constexpr = K * K * K
    x_b = x_ptr + b * IC * D * H * W

    for ic in range(0, IC):
        x_c = x_b + ic * D * H * W
        w_c = w_ptr + ic * OC * KKK
        for kd in tl.static_range(K):
            td = od + P - kd
            vd = (td >= 0) & ((td % S) == 0)
            idd = tl.where(td >= 0, td, 0) // S
            vd = vd & (idd < D)
            for kh in tl.static_range(K):
                th = oh + P - kh
                vh = (th >= 0) & ((th % S) == 0)
                ih = tl.where(th >= 0, th, 0) // S
                vh = vh & (ih < H)
                vdh = vd & vh
                for kw in tl.static_range(K):
                    tw = ow + P - kw
                    vw = (tw >= 0) & ((tw % S) == 0)
                    iw = tl.where(tw >= 0, tw, 0) // S
                    vw = vw & (iw < W) & ow_mask
                    xm = vw & vdh
                    xv = tl.load(x_c + idd * H * W + ih * W + iw, mask=xm, other=0.0).to(tl.float32)
                    wv = tl.load(w_c + oc * KKK + (kd * K + kh) * K + kw,
                                 mask=oc_mask, other=0.0).to(tl.float32)
                    acc += wv[:, None] * xv[None, :]

    if HAS_BIAS:
        bv = tl.load(b_ptr + oc, mask=oc_mask, other=0.0).to(tl.float32)
        acc += bv[:, None]

    # LayerNorm over last dim (W_out)
    wm2 = ow_mask[None, :]
    acc_m = tl.where(wm2, acc, 0.0)
    mean = tl.sum(acc_m, axis=1) / W_out
    diff = tl.where(wm2, acc - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / W_out
    rstd = 1.0 / tl.sqrt(var + eps)
    lnw = tl.load(lnw_ptr + ow, mask=ow_mask, other=0.0).to(tl.float32)
    lnb = tl.load(lnb_ptr + ow, mask=ow_mask, other=0.0).to(tl.float32)
    y = diff * rstd[:, None] * lnw[None, :] + lnb[None, :]

    # exact GELU
    y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
    y = y * scale

    out_base = out_ptr + b * OC * D_out * H_out * W_out + od * H_out * W_out + oh * W_out
    optrs = out_base + oc[:, None] * (D_out * H_out * W_out) + ow[None, :]
    omask = oc_mask[:, None] & wm2
    tl.store(optrs, y.to(out_ptr.dtype.element_ty), mask=omask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.contiguous()
        conv = self.conv_transpose
        w = conv.weight.contiguous()
        B, IC, D, H, W = x.shape
        OC = w.shape[1]
        K = conv.kernel_size[0]
        S = conv.stride[0]
        P = conv.padding[0]
        D_out = (D - 1) * S - 2 * P + K
        H_out = (H - 1) * S - 2 * P + K
        W_out = (W - 1) * S - 2 * P + K

        out = torch.empty((B, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

        has_bias = conv.bias is not None
        b_t = conv.bias.contiguous() if has_bias else w
        lnw = self.layer_norm.weight.contiguous()
        lnb = self.layer_norm.bias.contiguous()

        BLOCK_OC = 32
        BLOCK_W = triton.next_power_of_2(W_out)
        grid = (B * D_out * H_out, triton.cdiv(OC, BLOCK_OC))
        convt3d_ln_gelu_kernel[grid](
            x, w, b_t, lnw, lnb, out,
            IC, OC, D, H, W, D_out, H_out, W_out,
            float(self.layer_norm.eps), float(self.scaling_factor),
            S=S, P=P, K=K, HAS_BIAS=has_bias,
            BLOCK_OC=BLOCK_OC, BLOCK_W=BLOCK_W,
            num_warps=4,
        )
        return out
