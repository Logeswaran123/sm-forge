import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def convt_pool_partial_kernel(
    x_ptr, w_ptr, b_ptr, part_ptr,
    Cin, Cout, H, W, PH, PW, NT,
    ht_min, ht_max,
    KS: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
    MK: tl.constexpr, MS: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_co = tl.program_id(2)

    P = PH * PW
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = offs_p < P
    ph = offs_p // PW
    pw = offs_p % PW

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_valid = offs_co < Cout
    offs_ci_base = tl.arange(0, BLOCK_CI)

    x_base = x_ptr + pid_n.to(tl.int64) * Cin * H * W
    HW = H * W

    best = tl.full((BLOCK_P, BLOCK_CO), float("-inf"), dtype=tl.float32)

    for dy in tl.static_range(MK):
        for dx in tl.static_range(MK):
            oh = ph * MS + dy
            ow = pw * MS + dx
            acc = tl.zeros((BLOCK_P, BLOCK_CO), dtype=tl.float32)
            for kh in tl.static_range(KS):
                for kw in tl.static_range(KS):
                    ih_num = oh + PAD - kh
                    iw_num = ow + PAD - kw
                    ih_nn = tl.where(ih_num >= 0, ih_num, 0)
                    iw_nn = tl.where(iw_num >= 0, iw_num, 0)
                    ih = ih_nn // STRIDE
                    iw = iw_nn // STRIDE
                    pix_ok = (ih_num >= 0) & (iw_num >= 0) \
                        & ((ih_nn % STRIDE) == 0) & ((iw_nn % STRIDE) == 0) \
                        & (ih < H) & (iw < W) & p_valid
                    pix_off = ih * W + iw
                    for ci0 in range(0, Cin, BLOCK_CI):
                        offs_ci = ci0 + offs_ci_base
                        ci_valid = offs_ci < Cin
                        a_ptrs = x_base + offs_ci[None, :] * HW + pix_off[:, None]
                        a = tl.load(a_ptrs, mask=pix_ok[:, None] & ci_valid[None, :], other=0.0)
                        w_ptrs = w_ptr + offs_ci[:, None] * (Cout * KS * KS) \
                            + offs_co[None, :] * (KS * KS) + (kh * KS + kw)
                        bw = tl.load(w_ptrs, mask=ci_valid[:, None] & co_valid[None, :], other=0.0)
                        acc = tl.dot(a.to(tl.float16), bw.to(tl.float16), acc)
            best = tl.maximum(best, acc)

    bias = tl.load(b_ptr + offs_co, mask=co_valid, other=0.0).to(tl.float32)
    v = best + bias[None, :]
    v = tl.minimum(tl.maximum(v, ht_min), ht_max)
    v = tl.where(p_valid[:, None], v, 0.0)
    s = tl.sum(v, axis=0)
    out_ptrs = part_ptr + (pid_n * NT + pid_p) * Cout + offs_co
    tl.store(out_ptrs, s, mask=co_valid)


@triton.jit
def finalize_kernel(part_ptr, out_ptr, NT, Cout, inv_count,
                    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_valid = offs_c < Cout
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for t0 in range(0, NT, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        m = (offs_t[:, None] < NT) & c_valid[None, :]
        ptrs = part_ptr + (pid_n * NT + offs_t[:, None]) * Cout + offs_c[None, :]
        acc += tl.load(ptrs, mask=m, other=0.0)
    s = tl.sum(acc, axis=0) * inv_count
    # tanh(s) = 1 - 2 / (exp(2s) + 1), numerically safe for large |s|
    e = tl.exp(2.0 * s)
    t = 1.0 - 2.0 / (e + 1.0)
    tl.store(out_ptr + pid_n * Cout + offs_c, t.to(out_ptr.dtype.element_ty), mask=c_valid)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        self.mk = maxpool_kernel_size
        self.ms = maxpool_stride
        self.ht_min = float(hardtanh_min)
        self.ht_max = float(hardtanh_max)

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        N, Cin, H, W = x.shape
        Cout = w.shape[1]
        KS, S, PAD = self.ks, self.stride, self.padding
        OH = (H - 1) * S - 2 * PAD + KS
        OW = (W - 1) * S - 2 * PAD + KS
        PH = (OH - self.mk) // self.ms + 1
        PW = (OW - self.mk) // self.ms + 1
        P = PH * PW

        BLOCK_P, BLOCK_CI, BLOCK_CO = 64, 32, 64
        NT = triton.cdiv(P, BLOCK_P)
        part = torch.empty((N, NT, Cout), device=x.device, dtype=torch.float32)
        grid = (N, NT, triton.cdiv(Cout, BLOCK_CO))
        convt_pool_partial_kernel[grid](
            x, w, b, part,
            Cin, Cout, H, W, PH, PW, NT,
            self.ht_min, self.ht_max,
            KS=KS, STRIDE=S, PAD=PAD, MK=self.mk, MS=self.ms,
            BLOCK_P=BLOCK_P, BLOCK_CI=BLOCK_CI, BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )

        out = torch.empty((N, Cout, 1, 1), device=x.device, dtype=x.dtype)
        BLOCK_C = 64
        finalize_kernel[(N, triton.cdiv(Cout, BLOCK_C))](
            part, out, NT, Cout, 1.0 / float(P),
            BLOCK_T=32, BLOCK_C=BLOCK_C, num_warps=4,
        )
        return out
