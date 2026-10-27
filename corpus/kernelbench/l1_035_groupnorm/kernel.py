import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gn_partial_stats(x_ptr, pcount_ptr, pmean_ptr, pm2_ptr,
                      group_size, num_chunks,
                      CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    gid = tl.program_id(0)
    cid = tl.program_id(1)
    base = gid.to(tl.int64) * group_size
    start = cid.to(tl.int64) * CHUNK
    offs = tl.arange(0, BLOCK)
    acc_s = tl.zeros([BLOCK], dtype=tl.float32)
    acc_q = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, CHUNK, BLOCK):
        idx = start + i + offs
        m = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
        acc_s += v
        acc_q += v * v
    s = tl.sum(acc_s, axis=0)
    q = tl.sum(acc_q, axis=0)
    remaining = group_size - start
    cnt = tl.minimum(remaining, CHUNK).to(tl.float32)
    cnt = tl.maximum(cnt, 1.0)
    mean = s / cnt
    m2 = q - s * mean
    m2 = tl.maximum(m2, 0.0)
    pidx = gid * num_chunks + cid
    tl.store(pcount_ptr + pidx, cnt)
    tl.store(pmean_ptr + pidx, mean)
    tl.store(pm2_ptr + pidx, m2)


@triton.jit
def _gn_combine(pcount_ptr, pmean_ptr, pm2_ptr, mean_ptr, rstd_ptr,
                num_chunks, group_size, eps,
                BLOCK: tl.constexpr):
    gid = tl.program_id(0)
    base = gid * num_chunks
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, num_chunks, BLOCK):
        idx = i + offs
        m = idx < num_chunks
        c = tl.load(pcount_ptr + base + idx, mask=m, other=0.0)
        mu = tl.load(pmean_ptr + base + idx, mask=m, other=0.0)
        acc += c * mu
    n = group_size.to(tl.float32)
    mean = tl.sum(acc, axis=0) / n
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, num_chunks, BLOCK):
        idx = i + offs
        m = idx < num_chunks
        c = tl.load(pcount_ptr + base + idx, mask=m, other=0.0)
        mu = tl.load(pmean_ptr + base + idx, mask=m, other=0.0)
        m2 = tl.load(pm2_ptr + base + idx, mask=m, other=0.0)
        d = mu - mean
        acc2 += tl.where(m, m2 + c * d * d, 0.0)
    var = tl.sum(acc2, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + gid, mean)
    tl.store(rstd_ptr + gid, rstd)


@triton.jit
def _gn_normalize(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr,
                  group_size, HW, G, CPG,
                  BLOCK: tl.constexpr):
    gid = tl.program_id(0)
    bid = tl.program_id(1)
    g = gid % G
    base = gid.to(tl.int64) * group_size
    idx = bid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = idx < group_size
    mean = tl.load(mean_ptr + gid)
    rstd = tl.load(rstd_ptr + gid)
    ch = g * CPG + (idx // HW).to(tl.int32)
    w = tl.load(w_ptr + ch, mask=m, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + ch, mask=m, other=0.0).to(tl.float32)
    v = tl.load(x_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
    out = (v - mean) * rstd * w + b
    tl.store(y_ptr + base + idx, out.to(y_ptr.dtype.element_ty), mask=m)


class Model(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        N = x.shape[0]
        C = x.shape[1]
        G = self.gn.num_groups
        CPG = C // G
        HW = 1
        for s in x.shape[2:]:
            HW *= s
        group_size = CPG * HW
        NG = N * G

        CHUNK = 65536
        BLOCK1 = 4096
        num_chunks = triton.cdiv(group_size, CHUNK)
        dev = x.device
        pcount = torch.empty(NG * num_chunks, device=dev, dtype=torch.float32)
        pmean = torch.empty(NG * num_chunks, device=dev, dtype=torch.float32)
        pm2 = torch.empty(NG * num_chunks, device=dev, dtype=torch.float32)
        mean = torch.empty(NG, device=dev, dtype=torch.float32)
        rstd = torch.empty(NG, device=dev, dtype=torch.float32)

        _gn_partial_stats[(NG, num_chunks)](
            x, pcount, pmean, pm2, group_size, num_chunks,
            CHUNK=CHUNK, BLOCK=BLOCK1, num_warps=8)
        _gn_combine[(NG,)](
            pcount, pmean, pm2, mean, rstd, num_chunks, group_size,
            float(self.gn.eps), BLOCK=256, num_warps=4)

        y = torch.empty_like(x)
        BLOCK3 = 4096
        _gn_normalize[(NG, triton.cdiv(group_size, BLOCK3))](
            x, y, self.gn.weight, self.gn.bias, mean, rstd,
            group_size, HW, G, CPG, BLOCK=BLOCK3, num_warps=8)
        return y
