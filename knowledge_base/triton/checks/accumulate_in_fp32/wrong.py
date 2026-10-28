import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _splitk_matmul_kernel(
    a_ptr, b_ptr, ws_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    K_PER_SPLIT,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    k_start = pid_k * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_start + kk + offs_kb
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        )
        acc = (acc.to(tl.float32) + tl.dot(a, b)).to(tl.bfloat16)

    ws_offs = pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(ws_ptr + ws_offs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _splitk_reduce_kernel(
    ws_ptr, c_ptr,
    MN, SPLIT,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, SPLIT):
        acc += tl.load(ws_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty), mask=mask)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
        MAX_SPLIT = 1

        n_kblocks = triton.cdiv(K, BLOCK_K)
        split = max(1, min(MAX_SPLIT, n_kblocks))
        k_per_split = triton.cdiv(n_kblocks, split) * BLOCK_K
        split = triton.cdiv(K, k_per_split)

        ws = torch.empty((split, M, N), device=A.device, dtype=torch.float32)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), split)
        _splitk_matmul_kernel[grid](
            A, B, ws,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            k_per_split,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        MN = M * N
        RBLOCK = 1024
        _splitk_reduce_kernel[(triton.cdiv(MN, RBLOCK),)](
            ws, C, MN, split,
            BLOCK=RBLOCK, num_warps=4,
        )
        return C


M = 256
N = 256
K = 131072 * 4


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []
