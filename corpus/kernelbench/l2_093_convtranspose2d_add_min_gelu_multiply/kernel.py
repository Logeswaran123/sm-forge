import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    C_in, H, W, C_out, OH, OW,
    add_value, mul_value,
    STRIDE: tl.constexpr, PAD: tl.constexpr, K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_oc = tl.program_id(1)
    n = tl.program_id(2)

    OHW = OH * OW
    HW = H * W

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, BLOCK_IC)

    p_mask = offs_p < OHW
    oc_mask = offs_oc < C_out

    oh = offs_p // OW
    ow = offs_p % OW

    x_base = x_ptr + n.to(tl.int64) * C_in * HW

    acc = tl.zeros((BLOCK_P, BLOCK_OC), dtype=tl.float32)

    for kh in range(K):
        th = oh + PAD - kh
        th_c = tl.where(th >= 0, th, 0)
        ih = th_c // STRIDE
        vh = (th >= 0) & ((th_c % STRIDE) == 0) & (ih < H)
        for kw in range(K):
            tw = ow + PAD - kw
            tw_c = tl.where(tw >= 0, tw, 0)
            iw = tw_c // STRIDE
            vw = (tw >= 0) & ((tw_c % STRIDE) == 0) & (iw < W)
            valid = vh & vw & p_mask
            sp_off = tl.where(valid, ih * W + iw, 0)
            for ic0 in range(0, C_in, BLOCK_IC):
                ic = ic0 + offs_ic
                ic_mask = ic < C_in
                # A tile: [BLOCK_P, BLOCK_IC]
                a_ptrs = x_base + ic[None, :] * HW + sp_off[:, None]
                a = tl.load(a_ptrs, mask=valid[:, None] & ic_mask[None, :], other=0.0)
                # B tile: w[ic, oc, kh, kw] -> [BLOCK_IC, BLOCK_OC]
                b_ptrs = w_ptr + ic[:, None] * (C_out * K * K) + offs_oc[None, :] * (K * K) + kh * K + kw
                b = tl.load(b_ptrs, mask=ic_mask[:, None] & oc_mask[None, :], other=0.0)
                acc += tl.dot(a.to(tl.float16), b.to(tl.float16), out_dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    v = acc + add_value
    v = tl.minimum(v, 0.0)
    v = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
    v = v * mul_value

    out_base = out_ptr + n.to(tl.int64) * C_out * OHW
    o_ptrs = out_base + offs_oc[None, :] * OHW + offs_p[:, None]
    tl.store(o_ptrs, v.to(out_ptr.dtype.element_ty), mask=p_mask[:, None] & oc_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        conv = self.conv_transpose
        x = x.contiguous()
        w = conv.weight.contiguous()
        b = conv.bias
        N, C_in, H, W = x.shape
        C_out = w.shape[1]
        K = w.shape[2]
        S = conv.stride[0]
        P = conv.padding[0]
        OP = conv.output_padding[0]
        OH = (H - 1) * S - 2 * P + K + OP
        OW = (W - 1) * S - 2 * P + K + OP

        out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_P, BLOCK_OC, BLOCK_IC = 64, 64, 32
        grid = (triton.cdiv(OH * OW, BLOCK_P), triton.cdiv(C_out, BLOCK_OC), N)
        has_bias = b is not None
        b_arg = b.contiguous() if has_bias else w
        convt_fused_kernel[grid](
            x, w, b_arg, out,
            C_in, H, W, C_out, OH, OW,
            float(self.add_value), float(self.multiply_value),
            STRIDE=S, PAD=P, K=K, HAS_BIAS=has_bias,
            BLOCK_P=BLOCK_P, BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC,
            num_warps=4,
        )
        return out
