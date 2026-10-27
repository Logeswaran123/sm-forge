import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tanh(y):
    # stable: exp overflow -> inf -> 1; underflow -> 0 -> -1
    return 1.0 - 2.0 / (tl.exp(2.0 * y) + 1.0)


@triton.jit
def conv3d_mish_tanh_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    NB, Cin, D, H, W,
    Cout, OD, OH, OW,
    stride, pad,
    KS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = NB * OD * OH * OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_m64 = offs_m.to(tl.int64)
    ow = offs_m64 % OW
    t = offs_m64 // OW
    oh = t % OH
    t = t // OH
    od = t % OD
    n = t // OD

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < Cout
    offs_n64 = offs_n.to(tl.int64)

    DHW = D.to(tl.int64) * H * W
    KVOL: tl.constexpr = KS * KS * KS
    Ktot = Cin * KVOL

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)

    for kd in tl.static_range(KS):
        idd = od * stride - pad + kd
        vd = (idd >= 0) & (idd < D)
        for kh in tl.static_range(KS):
            ih = oh * stride - pad + kh
            vh = (ih >= 0) & (ih < H)
            for kw in tl.static_range(KS):
                iw = ow * stride - pad + kw
                vw = (iw >= 0) & (iw < W)
                valid = m_mask & vd & vh & vw
                base = n * Cin * DHW + (idd * H + ih) * W + iw
                kpos = (kd * KS + kh) * KS + kw
                for c0 in range(0, Cin, BLOCK_K):
                    ci = c0 + offs_k
                    c_mask = ci < Cin
                    ci64 = ci.to(tl.int64)
                    a_ptrs = x_ptr + base[:, None] + ci64[None, :] * DHW
                    a = tl.load(a_ptrs, mask=valid[:, None] & c_mask[None, :], other=0.0)
                    b_ptrs = w_ptr + offs_n64[None, :] * Ktot + ci64[:, None] * KVOL + kpos
                    b = tl.load(b_ptrs, mask=c_mask[:, None] & n_mask[None, :], other=0.0)
                    acc = tl.dot(a, b, acc)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    v = acc + bias[None, :]

    # mish: v * tanh(softplus(v)), softplus threshold 20 (PyTorch)
    sp = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(tl.minimum(v, 20.0))))
    mval = v * _tanh(sp)
    out = _tanh(mval)

    S = OD.to(tl.int64) * OH * OW
    sp_idx = (od * OH + oh) * OW + ow
    y_ptrs = y_ptr + (n * Cout * S + sp_idx)[:, None] + offs_n64[None, :] * S
    tl.store(y_ptrs, out.to(y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        conv = self.conv
        KS = conv.kernel_size[0]
        stride = conv.stride[0]
        pad = conv.padding[0]
        x = x.contiguous()
        NB, Cin, D, H, W = x.shape
        Cout = conv.out_channels
        OD = (D + 2 * pad - KS) // stride + 1
        OH = (H + 2 * pad - KS) // stride + 1
        OW = (W + 2 * pad - KS) // stride + 1

        w = conv.weight.to(x.dtype).contiguous()
        if conv.bias is not None:
            b = conv.bias.to(torch.float32).contiguous()
        else:
            b = torch.zeros(Cout, device=x.device, dtype=torch.float32)

        y = torch.empty((NB, Cout, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
        M = NB * OD * OH * OW
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(Cout, BLOCK_N))
        conv3d_mish_tanh_kernel[grid](
            x, w, b, y,
            NB, Cin, D, H, W,
            Cout, OD, OH, OW,
            stride, pad,
            KS=KS,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return y
