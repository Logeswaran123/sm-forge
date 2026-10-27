import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_act_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    Cin, D, H, W,
    Cout, OD, OH, OW,
    M_total, K_total,
    KS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M_total
    n_mask = offs_n < Cout

    # decompose output position
    ow = offs_m % OW
    t = offs_m // OW
    oh = t % OH
    t = t // OH
    od = t % OD
    nb = t // OD

    HW = H * W
    DHW = D * HW
    x_base = nb.to(tl.int64) * Cin * DHW + od * HW + oh * W + ow  # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    KK: tl.constexpr = KS * KS * KS
    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total
        ci = offs_k // KK
        r = offs_k % KK
        kd = r // (KS * KS)
        kh = (r // KS) % KS
        kw = r % KS
        k_off = ci * DHW + kd * HW + kh * W + kw  # [BLOCK_K]

        a_ptrs = x_ptr + x_base[:, None] + k_off[None, :]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        b_ptrs = w_ptr + offs_n[None, :] * K_total + offs_k[:, None]
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, b)

    cb = tl.load(cb_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    eb = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    v = acc + cb[None, :]
    # relu
    v = tl.maximum(v, 0.0)
    # leaky relu
    v = tl.where(v >= 0.0, v, v * 0.01)
    # gelu (exact)
    v = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
    # sigmoid
    v = 1.0 / (1.0 + tl.exp(-v))
    # bias
    v = v + eb[None, :]

    ODHW = OD * OH * OW
    sp = offs_m - nb * ODHW
    out_off = nb.to(tl.int64)[:, None] * Cout * ODHW + offs_n[None, :] * ODHW + sp[:, None]
    tl.store(out_ptr + out_off, v.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        N, Cin, D, H, W = x.shape
        w = self.conv.weight
        Cout = w.shape[0]
        KS = w.shape[2]
        OD, OH, OW = D - KS + 1, H - KS + 1, W - KS + 1
        K_total = Cin * KS * KS * KS
        M_total = N * OD * OH * OW

        w2 = w.reshape(Cout, K_total).to(x.dtype).contiguous()
        cb = self.conv.bias.contiguous()
        eb = self.bias.reshape(Cout).contiguous()

        out = torch.empty((N, Cout, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M = 128
        BLOCK_N = max(16, min(64, triton.next_power_of_2(Cout)))
        BLOCK_K = 32
        grid = (triton.cdiv(M_total, BLOCK_M), triton.cdiv(Cout, BLOCK_N))
        conv3d_act_kernel[grid](
            x, w2, cb, eb, out,
            Cin, D, H, W,
            Cout, OD, OH, OW,
            M_total, K_total,
            KS=KS,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return out
