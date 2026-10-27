import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_partial_stats(x_ptr, cnt_ptr, mean_ptr, m2_ptr, N, num_chunks,
                      BLOCK: tl.constexpr):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = row.to(tl.int64) * N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    cnt = tl.sum(mask.to(tl.float32), axis=0)
    mean = tl.sum(x, axis=0) / cnt
    d = tl.where(mask, x - mean, 0.0)
    m2 = tl.sum(d * d, axis=0)
    idx = row * num_chunks + chunk
    tl.store(cnt_ptr + idx, cnt)
    tl.store(mean_ptr + idx, mean)
    tl.store(m2_ptr + idx, m2)


@triton.jit
def _ln_combine_stats(cnt_ptr, mean_ptr, m2_ptr, out_mean_ptr, out_rstd_ptr,
                      num_chunks, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * num_chunks
    run_n = 0.0
    run_mean = 0.0
    run_m2 = 0.0
    for start in range(0, num_chunks, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < num_chunks
        c = tl.load(cnt_ptr + base + offs, mask=mask, other=0.0)
        m = tl.load(mean_ptr + base + offs, mask=mask, other=0.0)
        q = tl.load(m2_ptr + base + offs, mask=mask, other=0.0)
        nb = tl.sum(c, axis=0)
        mb = tl.sum(c * m, axis=0) / nb
        dm = tl.where(mask, m - mb, 0.0)
        qb = tl.sum(q, axis=0) + tl.sum(c * dm * dm, axis=0)
        # Chan combine
        n_new = run_n + nb
        delta = mb - run_mean
        run_mean = run_mean + delta * (nb / n_new)
        run_m2 = run_m2 + qb + delta * delta * (run_n * nb / n_new)
        run_n = n_new
    var = run_m2 / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(out_mean_ptr + row, run_mean)
    tl.store(out_rstd_ptr + row, rstd)


@triton.jit
def _ln_normalize(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, N,
                  HAS_W: tl.constexpr, HAS_B: tl.constexpr,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = row.to(tl.int64) * N
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd
    if HAS_W:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    if HAS_B:
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y + b
    tl.store(y_ptr + base + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ns = self.ln.normalized_shape
        N = 1
        for s in ns:
            N *= s
        x_c = x.contiguous()
        M = x_c.numel() // N
        x2 = x_c.view(M, N)
        y = torch.empty_like(x2)

        STAT_BLOCK = 4096
        num_chunks = triton.cdiv(N, STAT_BLOCK)
        dev = x.device
        cnt = torch.empty((M, num_chunks), device=dev, dtype=torch.float32)
        pmean = torch.empty((M, num_chunks), device=dev, dtype=torch.float32)
        pm2 = torch.empty((M, num_chunks), device=dev, dtype=torch.float32)
        mean = torch.empty((M,), device=dev, dtype=torch.float32)
        rstd = torch.empty((M,), device=dev, dtype=torch.float32)

        _ln_partial_stats[(num_chunks, M)](x2, cnt, pmean, pm2, N, num_chunks,
                                           BLOCK=STAT_BLOCK, num_warps=8)
        _ln_combine_stats[(M,)](cnt, pmean, pm2, mean, rstd, num_chunks, N,
                                float(self.ln.eps), BLOCK=1024, num_warps=4)

        w = self.ln.weight
        b = self.ln.bias
        HAS_W = w is not None
        HAS_B = b is not None
        w_arg = w.contiguous().view(-1) if HAS_W else mean
        b_arg = b.contiguous().view(-1) if HAS_B else mean
        NORM_BLOCK = 4096
        _ln_normalize[(triton.cdiv(N, NORM_BLOCK), M)](
            x2, y, w_arg, b_arg, mean, rstd, N,
            HAS_W=HAS_W, HAS_B=HAS_B, BLOCK=NORM_BLOCK, num_warps=8)
        return y.view(x.shape)
