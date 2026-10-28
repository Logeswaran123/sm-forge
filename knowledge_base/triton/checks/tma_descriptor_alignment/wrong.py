import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def matmul_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, k0])
        b = b_desc.load([k0, pid_n * BLOCK_N])
        acc = tl.dot(a, b, acc)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(c_desc.dtype))


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        _, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N])
        c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        matmul_kernel[grid](
            a_desc, b_desc, c_desc,
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=8,
            num_warps=4, num_stages=3,
        )
        return C
