import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avgpool_cl_kernel(x_ptr, out_ptr, total, C, D, H, W, Dp, Hp, Wp,
                      P: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    o = offs.to(tl.int64)
    # output layout [B, Dp, Hp, Wp, C]
    c = o % C
    t = o // C
    pw = t % Wp
    t = t // Wp
    ph = t % Hp
    t = t // Hp
    pd = t % Dp
    b = t // Dp
    base = ((b * C + c) * D + pd * P) * H * W + (ph * P) * W + pw * P
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for kd in tl.static_range(P):
        for kh in tl.static_range(P):
            for kw in tl.static_range(P):
                v = tl.load(x_ptr + base + (kd * H + kh) * W + kw, mask=mask, other=0.0)
                acc += v.to(tl.float32)
    acc = acc / (P * P * P)
    tl.store(out_ptr + o, acc.to(tl.float16), mask=mask)


@triton.jit
def convt_kernel(p_ptr, w_ptr, bias_ptr, y_ptr,
                 Cin, Cout, Din, Hin, Win, Dout, Hout, Wout,
                 cmin, cmax,
                 KS: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    S = Dout * Hout * Wout

    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < S
    ow = offs_m % Wout
    t = offs_m // Wout
    oh = t % Hout
    od = t // Hout

    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < Cout

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kd in tl.static_range(KS):
        td = od + PAD - kd
        idd = td // STRIDE
        vd = (td >= 0) & ((td % STRIDE) == 0) & (idd < Din)
        for kh in tl.static_range(KS):
            th = oh + PAD - kh
            ih = th // STRIDE
            vh = (th >= 0) & ((th % STRIDE) == 0) & (ih < Hin)
            for kw in tl.static_range(KS):
                tw = ow + PAD - kw
                iw = tw // STRIDE
                vw = (tw >= 0) & ((tw % STRIDE) == 0) & (iw < Win)
                valid = vd & vh & vw & m_mask
                row = ((b * Din + idd) * Hin + ih) * Win + iw
                row = tl.where(valid, row, 0)
                tap = (kd * KS + kh) * KS + kw
                for ci0 in range(0, Cin, BK):
                    ci = ci0 + tl.arange(0, BK)
                    ci_mask = ci < Cin
                    a = tl.load(p_ptr + row[:, None] * Cin + ci[None, :],
                                mask=valid[:, None] & ci_mask[None, :], other=0.0)
                    wv = tl.load(w_ptr + (tap * Cin + ci)[:, None] * Cout + offs_n[None, :],
                                 mask=ci_mask[:, None] & n_mask[None, :], other=0.0)
                    acc = tl.dot(a, wv, acc)

    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    acc = tl.minimum(tl.maximum(acc, cmin), cmax)
    y_off = (b * Cout + offs_n.to(tl.int64))[None, :] * S + offs_m[:, None]
    tl.store(y_ptr + y_off, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def softmax_stats_kernel(y_ptr, max_ptr, sum_ptr, S, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * S
    m = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    s = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, S, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < S
        x = tl.load(y_ptr + base + offs, mask=mask, other=float("-inf"))
        m_new = tl.maximum(m, x)
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m - m_new))
        e = tl.where(mask, tl.exp(x - m_new), 0.0)
        s = s * alpha + e
        m = m_new
    M = tl.max(m, axis=0)
    scale = tl.where(m == float("-inf"), 0.0, tl.exp(m - M))
    total = tl.sum(s * scale, axis=0)
    tl.store(max_ptr + row, M)
    tl.store(sum_ptr + row, total)


@triton.jit
def softmax_norm_kernel(y_ptr, max_ptr, sum_ptr, scale_ptr, out_ptr, S, C,
                        BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    base = row.to(tl.int64) * S
    M = tl.load(max_ptr + row)
    tot = tl.load(sum_ptr + row)
    c = row % C
    sc = tl.load(scale_ptr + c).to(tl.float32)
    x = tl.load(y_ptr + base + offs, mask=mask, other=0.0)
    o = tl.exp(x - M) / tot * sc
    tl.store(out_ptr + base + offs, o.to(out_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding,
                                                 output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool = pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        B, Cin, D, H, W = x.shape
        P = self.pool
        Dp, Hp, Wp = D // P, H // P, W // P
        dev = x.device

        # 1) avg pool -> channels-last fp16
        pooled = torch.empty((B, Dp, Hp, Wp, Cin), device=dev, dtype=torch.float16)
        total = pooled.numel()
        BLOCK = 1024
        avgpool_cl_kernel[(triton.cdiv(total, BLOCK),)](
            x, pooled, total, Cin, D, H, W, Dp, Hp, Wp, P=P, BLOCK=BLOCK)

        # 2) transposed conv + bias + clamp
        KS, ST, PD, OP = self.ks, self.stride, self.padding, self.output_padding
        Cout = self.out_channels
        Dout = (Dp - 1) * ST - 2 * PD + KS + OP
        Hout = (Hp - 1) * ST - 2 * PD + KS + OP
        Wout = (Wp - 1) * ST - 2 * PD + KS + OP
        S = Dout * Hout * Wout

        w = self.conv_transpose.weight  # [Cin, Cout, kd, kh, kw]
        wt = w.permute(2, 3, 4, 0, 1).contiguous().to(torch.float16)  # [kd,kh,kw,Cin,Cout]
        bias = self.conv_transpose.bias.contiguous().to(torch.float32)

        y = torch.empty((B, Cout, S), device=dev, dtype=torch.float32)
        BM, BN, BK = 64, 64, 32
        grid = (triton.cdiv(S, BM), triton.cdiv(Cout, BN), B)
        convt_kernel[grid](pooled, wt, bias, y,
                           Cin, Cout, Dp, Hp, Wp, Dout, Hout, Wout,
                           float(self.clamp_min), float(self.clamp_max),
                           KS=KS, STRIDE=ST, PAD=PD, BM=BM, BN=BN, BK=BK,
                           num_warps=4)

        # 3) softmax over spatial dims, times scale
        rows = B * Cout
        rmax = torch.empty((rows,), device=dev, dtype=torch.float32)
        rsum = torch.empty((rows,), device=dev, dtype=torch.float32)
        softmax_stats_kernel[(rows,)](y, rmax, rsum, S, BLOCK=2048, num_warps=8)

        out = torch.empty((B, Cout, Dout, Hout, Wout), device=dev, dtype=x.dtype)
        scale = self.scale.reshape(-1).contiguous()
        NB = 2048
        softmax_norm_kernel[(triton.cdiv(S, NB), rows)](
            y, rmax, rsum, scale, out, S, Cout, BLOCK=NB, num_warps=8)
        return out
