import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_convt_pool_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    CI, CO, D, H, W, OD2, OH2, OW2,
    S: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    WIN: tl.constexpr, A: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    fw = pid % OW2
    t = pid // OW2
    fh = t % OH2
    t = t // OH2
    fd = t % OD2
    n = t // OD2

    m = tl.arange(0, BLOCK_M)
    ma = m // (A * A)
    mb = (m // A) % A
    mc = m % A
    row_base_ok = m < A * A * A

    co = tl.arange(0, BLOCK_CO)
    co_ok = co < CO
    kk = tl.arange(0, BLOCK_K)

    HW = H * W
    DHW = D * HW
    x_base = x_ptr + n.to(tl.int64) * CI * DHW

    run_max = tl.full([BLOCK_CO], float("-inf"), tl.float32)

    for rd in range(S):
        for rh in range(S):
            for rw in range(S):
                od = WIN * fd + rd + S * ma
                oh = WIN * fh + rh + S * mb
                ow = WIN * fw + rw + S * mc
                row_ok = row_base_ok & ((rd + S * ma) < WIN) & ((rh + S * mb) < WIN) & ((rw + S * mc) < WIN)
                acc = tl.zeros([BLOCK_M, BLOCK_CO], dtype=tl.float32)
                for kd in range(K):
                    sd = WIN * fd + rd + P - kd + S * K
                    if sd % S == 0:
                        idd = (od + P - kd + S * K) // S - K
                        okd = row_ok & (idd >= 0) & (idd < D)
                        for kh in range(K):
                            sh = WIN * fh + rh + P - kh + S * K
                            if sh % S == 0:
                                ih = (oh + P - kh + S * K) // S - K
                                okh = okd & (ih >= 0) & (ih < H)
                                for kw in range(K):
                                    sw = WIN * fw + rw + P - kw + S * K
                                    if sw % S == 0:
                                        iw = (ow + P - kw + S * K) // S - K
                                        okw = okh & (iw >= 0) & (iw < W)
                                        sp_off = idd * HW + ih * W + iw
                                        tap = (kd * K + kh) * K + kw
                                        for c0 in range(0, CI, BLOCK_K):
                                            ci = c0 + kk
                                            ci_ok = ci < CI
                                            a_ptrs = x_base + ci[None, :] * DHW + sp_off[:, None]
                                            a = tl.load(a_ptrs, mask=okw[:, None] & ci_ok[None, :], other=0.0)
                                            w_ptrs = w_ptr + tap * CI * CO + ci[:, None] * CO + co[None, :]
                                            b = tl.load(w_ptrs, mask=ci_ok[:, None] & co_ok[None, :], other=0.0)
                                            acc = tl.dot(a, b, acc)
                acc = tl.where(row_ok[:, None], acc, float("-inf"))
                run_max = tl.maximum(run_max, tl.max(acc, axis=0))

    bias = tl.load(b_ptr + co, mask=co_ok, other=0.0).to(tl.float32)
    val = run_max + bias
    val = tl.where(co_ok, val, 0.0)
    s = tl.sum(val, axis=0)
    tl.store(out_ptr + pid, s.to(out_ptr.dtype.element_ty))


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.s = stride
        self.p = padding

    def forward(self, x):
        x = x.contiguous()
        N, CI, D, H, W = x.shape
        CO = self.out_channels
        K, S, P = self.k, self.s, self.p
        Do = (D - 1) * S - 2 * P + K
        Ho = (H - 1) * S - 2 * P + K
        Wo = (W - 1) * S - 2 * P + K
        OD2 = (Do // 2) // 3
        OH2 = (Ho // 2) // 3
        OW2 = (Wo // 2) // 3

        dt = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
        xin = x.to(dt).contiguous()
        # weight [CI, CO, K, K, K] -> [K^3, CI, CO]
        w = self.conv_transpose.weight.to(dt).permute(2, 3, 4, 0, 1).contiguous()
        b = self.conv_transpose.bias.to(torch.float32).contiguous()

        out = torch.empty((N, 1, OD2, OH2, OW2), device=x.device, dtype=x.dtype)
        total = N * OD2 * OH2 * OW2
        if total == 0:
            return out

        WIN = 6
        A = (WIN + S - 1) // S
        BLOCK_M = max(16, triton.next_power_of_2(A * A * A))
        BLOCK_K = max(16, min(64, triton.next_power_of_2(CI)))
        BLOCK_CO = max(16, triton.next_power_of_2(CO))

        grid = (total,)
        fused_convt_pool_sum_kernel[grid](
            xin, w, b, out,
            CI, CO, D, H, W, OD2, OH2, OW2,
            S=S, P=P, K=K, WIN=WIN, A=A,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_CO=BLOCK_CO,
            num_warps=4,
        )
        return out
