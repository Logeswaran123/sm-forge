import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt_fused_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, H_out, W_out,
    scale,
    STRIDE: tl.constexpr, PAD: tl.constexpr, K: tl.constexpr,
    HAS_CB: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_co = pid_c * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = tl.arange(0, BLOCK_CI)

    HWo = H_out * W_out
    total_p = N * HWo
    p_mask = offs_p < total_p
    co_mask = offs_co < C_out

    n = offs_p // HWo
    rem = offs_p % HWo
    oh = rem // W_out
    ow = rem % W_out
    n64 = n.to(tl.int64)

    HW = H * W
    acc = tl.zeros((BLOCK_P, BLOCK_CO), dtype=tl.float32)
    in_ty = x_ptr.dtype.element_ty

    for kh in tl.static_range(K):
        ih_num = oh + PAD - kh
        ih_ok = (ih_num >= 0)
        ih_nn = tl.where(ih_ok, ih_num, 0)
        ih_ok = ih_ok & ((ih_nn % STRIDE) == 0)
        ih = ih_nn // STRIDE
        ih_ok = ih_ok & (ih < H)
        for kw in tl.static_range(K):
            iw_num = ow + PAD - kw
            iw_ok = (iw_num >= 0)
            iw_nn = tl.where(iw_ok, iw_num, 0)
            iw_ok = iw_ok & ((iw_nn % STRIDE) == 0)
            iw = iw_nn // STRIDE
            iw_ok = iw_ok & (iw < W)

            pix_ok = p_mask & ih_ok & iw_ok
            pix_off = n64 * C_in * HW + (ih * W + iw).to(tl.int64)

            for ci0 in range(0, C_in, BLOCK_CI):
                ci = ci0 + offs_ci
                ci_mask = ci < C_in
                x_ptrs = x_ptr + pix_off[:, None] + (ci.to(tl.int64) * HW)[None, :]
                xt = tl.load(x_ptrs, mask=pix_ok[:, None] & ci_mask[None, :], other=0.0)
                # weight layout: [C_in, C_out, K, K]
                w_ptrs = w_ptr + ((ci[:, None] * C_out + offs_co[None, :]) * K + kh) * K + kw
                wt = tl.load(w_ptrs, mask=ci_mask[:, None] & co_mask[None, :], other=0.0)
                acc = tl.dot(xt.to(in_ty), wt.to(in_ty), acc, out_dtype=tl.float32)

    bias = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    if HAS_CB:
        bias += tl.load(cb_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    y = acc + bias[None, :]
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    y = y * scale
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    y = y / scale

    out_off = n64[:, None] * C_out * HWo + (offs_co.to(tl.int64) * HWo)[None, :] \
        + (oh * W_out + ow).to(tl.int64)[:, None]
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty),
             mask=p_mask[:, None] & co_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                                 padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        ct = self.conv_transpose
        x = x.contiguous()
        N, C_in, H, W = x.shape
        w = ct.weight.contiguous()
        C_out = w.shape[1]
        K = ct.kernel_size[0]
        S = ct.stride[0]
        P = ct.padding[0]
        OP = ct.output_padding[0]
        H_out = (H - 1) * S - 2 * P + (K - 1) + OP + 1
        W_out = (W - 1) * S - 2 * P + (K - 1) + OP + 1

        out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
        b = self.bias.contiguous().view(-1)
        has_cb = ct.bias is not None
        cb = ct.bias.contiguous() if has_cb else b

        BLOCK_P, BLOCK_CO, BLOCK_CI = 64, 64, 32
        grid = (triton.cdiv(N * H_out * W_out, BLOCK_P), triton.cdiv(C_out, BLOCK_CO))
        convt_fused_kernel[grid](
            x, w, cb, b, out,
            N, C_in, H, W, C_out, H_out, W_out,
            float(self.scaling_factor),
            STRIDE=S, PAD=P, K=K, HAS_CB=has_cb,
            BLOCK_P=BLOCK_P, BLOCK_CO=BLOCK_CO, BLOCK_CI=BLOCK_CI,
            num_warps=4,
        )
        return out
