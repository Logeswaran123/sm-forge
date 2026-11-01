import torch
from torch import nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"

using T = cutlass::half_t;
using Gemm = cutlass::gemm::device::Gemm<
    T, cutlass::layout::RowMajor, T, cutlass::layout::ColumnMajor, T, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>, cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<T, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

int gemm_nt(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream) {
  Gemm op;
  Gemm::Arguments args({m, n, k}, {(const T*)a, k}, {(const T*)b, k}, {(T*)c, n}, {(T*)c, n},
                       {1.0f, 0.0f});
  return int(op(args, nullptr, stream));
}
"""

CPP_SRC = r"""
int gemm_nt(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream);

torch::Tensor matmul_t(torch::Tensor a, torch::Tensor b) {
  auto c = torch::empty({a.size(0), b.size(0)}, a.options());
  int status = gemm_nt(a.data_ptr(), b.data_ptr(), c.data_ptr(), a.size(0), b.size(0), a.size(1),
                       at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == 0, "CUTLASS status ", status);
  return c;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["matmul_t"])


class Model(nn.Module):
    def forward(self, a, b):
        return ext.matmul_t(a.contiguous(), b.contiguous())


def get_inputs():
    a = torch.randn(256, 128, dtype=torch.float16)
    return [a, torch.randn(192, 128, dtype=torch.float16)]


def get_init_inputs():
    return []
