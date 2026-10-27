import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_tanh_scale_bias_maxpool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    C_in, H, W, C_out, PH, PW,
    scale,
    KS: tl.constexpr, POOL: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    n = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = offs_p < PH * PW
    ph = offs_p // PW
    pw = offs_p % PW

    offs_c = pid_c * BLOCK_CO + tl.arange(0, BLOCK_CO)
    c_mask = offs_c < C_out

    cbias = tl.load(cb_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)

    x_base = x_ptr + n.to(tl.int64) * C_in * H * W
    w_row = offs_c * (C_in * KS * KS)

    result = tl.full((BLOCK_P, BLOCK_CO), float("-inf"), dtype=tl.float32)

    for dh in range(POOL):
        for dw in range(POOL):
            oh = ph * POOL + dh
            ow = pw * POOL + dw
            acc = tl.zeros((BLOCK_P, BLOCK_CO), dtype=tl.float32)
            for ci in range(C_in):
                x_ci = x_base + ci * H * W
                w_ci = w_ptr + w_row + ci * KS * KS
                for kh in tl.static_range(KS):
                    for kw in tl.static_range(KS):
                        xv = tl.load(x_ci + (oh + kh) * W + (ow + kw),
                                     mask=p_mask, other=0.0).to(tl.float32)
                        wv = tl.load(w_ci + kh * KS + kw,
                                     mask=c_mask, other=0.0).to(tl.float32)
                        acc += xv[:, None] * wv[None, :]
            v = acc + cbias[None, :]
            # tanh(v) = 1 - 2 / (exp(2v) + 1), saturates correctly at +-inf
            e = tl.exp(2.0 * v)
            t = 1.0 - 2.0 / (e + 1.0)
            y = t * scale + bias[None, :]
            result = tl.maximum(result, y)

    out_base = out_ptr + n.to(tl.int64) * C_out * PH * PW
    out_ptrs = out_base + offs_c[None, :] * (PH * PW) + offs_p[:, None]
    mask = p_mask[:, None] & c_mask[None, :]
    tl.store(out_ptrs, result.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, C_in, H, W = x.shape
        w = self.conv.weight.contiguous()
        C_out = w.shape[0]
        KS = w.shape[2]
        cb = self.conv.bias.contiguous()
        b = self.bias.reshape(-1).contiguous()

        P = self.pool_kernel_size
        if isinstance(P, (tuple, list)):
            P = P[0]
        OH = H - KS + 1
        OW = W - KS + 1
        PH = (OH - P) // P + 1
        PW = (OW - P) // P + 1

        out = torch.empty((N, C_out, PH, PW), device=x.device, dtype=x.dtype)

        BLOCK_P = 64
        BLOCK_CO = 64
        grid = (N, triton.cdiv(PH * PW, BLOCK_P), triton.cdiv(C_out, BLOCK_CO))
        conv_tanh_scale_bias_maxpool_kernel[grid](
            x, w, cb, b, out,
            C_in, H, W, C_out, PH, PW,
            float(self.scaling_factor),
            KS=KS, POOL=P,
            BLOCK_P=BLOCK_P, BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )
        return out
