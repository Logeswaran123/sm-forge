import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_mish_bn_kernel(
    x_ptr, w_ptr, bias_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, out_ptr,
    M, OC, K, OH, OW,
    sxn, sxc, sxh, sxw,
    eps,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    ow = offs_m % OW
    t = offs_m // OW
    oh = t % OH
    nb = t // OH
    a_base = nb * sxn + oh * sxh + ow * sxw

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        c = offs_k // (KH * KW)
        r = offs_k % (KH * KW)
        kh = r // KW
        kw = r % KW
        k_off = c * sxc + kh * sxh + kw * sxw
        a = tl.load(x_ptr + a_base[:, None] + k_off[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * K + offs_k[:, None],
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    v = acc + bias[None, :]

    # mish: x * tanh(softplus(x)); softplus threshold 20 -> returns x
    xc = tl.minimum(v, 20.0)
    e = tl.exp(xc)
    nn_ = e * (e + 2.0)
    th = nn_ / (nn_ + 2.0)
    th = tl.where(v > 20.0, 1.0, th)
    y = v * th

    mean = tl.load(mean_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    scale = gamma / tl.sqrt(var + eps)
    shift = beta - mean * scale
    y = y * scale[None, :] + shift[None, :]

    OHW = OH * OW
    out_off = (nb * OC * OHW + oh * OW + ow)[:, None] + (offs_n * OHW)[None, :]
    tl.store(out_ptr + out_off, y.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        w = self.conv.weight
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        K = C * KH * KW
        M = N * OH * OW

        w2 = w.to(x.dtype).contiguous().view(OC, K)
        bias = self.conv.bias
        if bias is None:
            bias = torch.zeros(OC, device=x.device, dtype=torch.float32)

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(OC, BLOCK_N))
        conv_mish_bn_kernel[grid](
            x, w2, bias,
            self.bn.running_mean, self.bn.running_var,
            self.bn.weight, self.bn.bias, out,
            M, OC, K, OH, OW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            float(self.bn.eps),
            KH=KH, KW=KW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=2,
        )
        return out
