import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemv_partial_kernel(
    A, B, P,
    M, K,
    stride_am, stride_ak, stride_bk,
    k_per_split,
    BM: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)

    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    rows64 = rows.to(tl.int64)

    k_start = pid_s * k_per_split
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BK), dtype=tl.float32)
    for k0 in range(0, k_per_split, BK):
        ks = k_start + k0 + offs_k
        kmask = ks < K
        ks64 = ks.to(tl.int64)
        a_ptrs = A + rows64[:, None] * stride_am + ks64[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=rmask[:, None] & kmask[None, :], other=0.0)
        b = tl.load(B + ks64 * stride_bk, mask=kmask, other=0.0)
        acc += a.to(tl.float32) * b.to(tl.float32)[None, :]

    s = tl.sum(acc, axis=1)
    tl.store(P + pid_s * M + rows, s, mask=rmask)


@triton.jit
def reduce_partials_kernel(
    P, C,
    M, SPLIT,
    stride_cm,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < M
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, SPLIT):
        acc += tl.load(P + s * M + rows, mask=rmask, other=0.0)
    tl.store(C + rows.to(tl.int64) * stride_cm,
             acc.to(C.dtype.element_ty), mask=rmask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        assert B.shape[0] == K and B.shape[1] == 1

        BM = 8
        BK = 512
        num_row_blocks = triton.cdiv(M, BM)
        num_k_blocks = triton.cdiv(K, BK)

        target_programs = 84 * 8
        split = max(1, target_programs // num_row_blocks)
        split = min(split, num_k_blocks)
        k_blocks_per_split = triton.cdiv(num_k_blocks, split)
        k_per_split = k_blocks_per_split * BK
        split = triton.cdiv(K, k_per_split)

        partials = torch.empty((split, M), device=A.device, dtype=torch.float32)
        C = torch.empty((M, 1), device=A.device, dtype=A.dtype)

        grid = (num_row_blocks, split)
        gemv_partial_kernel[grid](
            A, B, partials,
            M, K,
            A.stride(0), A.stride(1), B.stride(0),
            k_per_split,
            BM=BM, BK=BK,
            num_warps=4,
        )

        RBLOCK = 256
        reduce_partials_kernel[(triton.cdiv(M, RBLOCK),)](
            partials, C,
            M, split,
            C.stride(0),
            BLOCK=RBLOCK,
            num_warps=4,
        )
        return C


M = 256 * 8
K = 131072 * 8


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]


def get_init_inputs():
    return []
