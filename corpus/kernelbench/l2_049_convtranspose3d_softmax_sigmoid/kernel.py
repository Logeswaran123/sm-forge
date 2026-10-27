import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt3d_softmax_sigmoid_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, Cin, Cout,
    D, H, W,
    OD, OH, OW,
    STRIDE: tl.constexpr, PAD: tl.constexpr, KS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    ow = offs_m % OW
    t = offs_m // OW
    oh = t % OH
    t = t // OH
    od = t % OD
    n = t // OD

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < Cout

    DHW = D * H * W
    x_batch_base = n.to(tl.int64) * Cin * DHW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kd in tl.static_range(KS):
        td = od + PAD - kd
        tdc = tl.where(td >= 0, td, 0)
        idd = tdc // STRIDE
        vd = (td >= 0) & ((tdc % STRIDE) == 0) & (idd < D)
        for kh in tl.static_range(KS):
            th = oh + PAD - kh
            thc = tl.where(th >= 0, th, 0)
            ih = thc // STRIDE
            vh = (th >= 0) & ((thc % STRIDE) == 0) & (ih < H)
            for kw in tl.static_range(KS):
                tw = ow + PAD - kw
                twc = tl.where(tw >= 0, tw, 0)
                iw = twc // STRIDE
                vw = (tw >= 0) & ((twc % STRIDE) == 0) & (iw < W)

                valid = m_mask & vd & vh & vw
                sp_in = (idd * H + ih) * W + iw
                x_base = x_batch_base + sp_in
                wtap = (kd * KS + kh) * KS + kw

                for k0 in range(0, Cin, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    k_mask = offs_k < Cin
                    a = tl.load(
                        x_ptr + x_base[:, None] + offs_k[None, :].to(tl.int64) * DHW,
                        mask=valid[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    b = tl.load(
                        w_ptr + (wtap * Cin + offs_k)[:, None] * Cout + offs_n[None, :],
                        mask=k_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )
                    acc = tl.dot(a, b, acc)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    logits = tl.where(n_mask[None, :], acc, float("-inf"))
    mx = tl.max(logits, axis=1)
    e = tl.exp(logits - mx[:, None])
    e = tl.where(n_mask[None, :], e, 0.0)
    s = tl.sum(e, axis=1)
    p = e / s[:, None]
    y = 1.0 / (1.0 + tl.exp(-p))

    OSP = OD * OH * OW
    osp = (od * OH + oh) * OW + ow
    out_off = (n.to(tl.int64) * Cout * OSP + osp)[:, None] + offs_n[None, :].to(tl.int64) * OSP
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                                 padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv = self.conv_transpose
        x = x.contiguous()
        N, Cin, D, H, W = x.shape
        weight = conv.weight  # (Cin, Cout, kD, kH, kW)
        Cout = weight.shape[1]
        KS = weight.shape[2]
        s = conv.stride[0]
        p = conv.padding[0]
        op = conv.output_padding[0]

        OD = (D - 1) * s - 2 * p + KS + op
        OH = (H - 1) * s - 2 * p + KS + op
        OW = (W - 1) * s - 2 * p + KS + op

        w = weight.to(x.dtype).permute(2, 3, 4, 0, 1).contiguous()  # (kD,kH,kW,Cin,Cout)
        has_bias = conv.bias is not None
        b = conv.bias.contiguous() if has_bias else w

        out = torch.empty((N, Cout, OD, OH, OW), device=x.device, dtype=x.dtype)

        M = N * OD * OH * OW
        BLOCK_M = 64
        BLOCK_N = max(16, triton.next_power_of_2(Cout))
        BLOCK_K = min(64, max(16, triton.next_power_of_2(Cin)))
        grid = (triton.cdiv(M, BLOCK_M),)
        convt3d_softmax_sigmoid_kernel[grid](
            x, w, b, out,
            M, Cin, Cout,
            D, H, W,
            OD, OH, OW,
            STRIDE=s, PAD=p, KS=KS,
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return out


batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]
