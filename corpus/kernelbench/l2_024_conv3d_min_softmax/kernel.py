import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    D, H, W, OD, OH, OW, COUT,
    CIN: tl.constexpr, K: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    b = tl.program_id(1)

    P = OH * OW
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < P
    oh = offs_p // OW
    ow = offs_p % OW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < COUT

    bias = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    HW = H * W
    DHW = D * HW
    x_base = x_ptr + b.to(tl.int64) * CIN * DHW
    KVOL: tl.constexpr = CIN * K * K * K

    minv = tl.full([BLOCK_C, BLOCK_P], float("inf"), dtype=tl.float32)
    sp_off = oh * W + ow

    for od in range(0, OD):
        acc = tl.zeros([BLOCK_C, BLOCK_P], dtype=tl.float32)
        for ci in tl.static_range(CIN):
            for kd in tl.static_range(K):
                for kh in tl.static_range(K):
                    for kw in tl.static_range(K):
                        x_off = ci * DHW + (od + kd) * HW + sp_off + kh * W + kw
                        xv = tl.load(x_base + x_off, mask=mask_p, other=0.0).to(tl.float32)
                        w_idx = ((ci * K + kd) * K + kh) * K + kw
                        wv = tl.load(w_ptr + offs_c * KVOL + w_idx, mask=mask_c, other=0.0).to(tl.float32)
                        acc += wv[:, None] * xv[None, :]
        minv = tl.minimum(minv, acc)

    vals = minv + bias[:, None]
    vals = tl.where(mask_c[:, None], vals, float("-inf"))
    mx = tl.max(vals, axis=0)
    e = tl.exp(vals - mx[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    s = tl.sum(e, axis=0)
    res = e / s[None, :]

    out_base = out_ptr + b.to(tl.int64) * COUT * P
    out_ptrs = out_base + offs_c[:, None] * P + offs_p[None, :]
    tl.store(out_ptrs, res.to(out_ptr.dtype.element_ty),
             mask=mask_c[:, None] & mask_p[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        if self.dim != 2:
            raise NotImplementedError("only dim=2 (depth) is supported")
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        B, CIN, D, H, W = x.shape
        COUT = w.shape[0]
        K = w.shape[2]
        OD, OH, OW = D - K + 1, H - K + 1, W - K + 1
        out = torch.empty((B, COUT, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_C = max(triton.next_power_of_2(COUT), 2)
        BLOCK_P = 128
        grid = (triton.cdiv(OH * OW, BLOCK_P), B)
        conv3d_min_softmax_kernel[grid](
            x, w, bias, out,
            D, H, W, OD, OH, OW, COUT,
            CIN=CIN, K=K,
            BLOCK_C=BLOCK_C, BLOCK_P=BLOCK_P,
            num_warps=4,
        )
        return out
