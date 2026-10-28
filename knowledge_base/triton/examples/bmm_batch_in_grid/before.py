import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a_ptr + bid.to(tl.int64) * stride_ab
    b_base = b_ptr + bid.to(tl.int64) * stride_bb
    c_base = c_ptr + bid.to(tl.int64) * stride_cb

    a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        Bsz, M, K = A.shape
        _, _, N = B.shape
        C = torch.empty((Bsz, M, N), device=A.device, dtype=A.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
        for b in range(Bsz):
            bmm_kernel[grid](
                A[b], B[b], C[b],
                M, N, K,
                0, A.stride(1), A.stride(2),
                0, B.stride(1), B.stride(2),
                0, C.stride(1), C.stride(2),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                GROUP_M=8,
                num_warps=8, num_stages=3,
            )
        return C
