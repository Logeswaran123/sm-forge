import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"

// CUTLASS's SM80 GEMM: 128x128x32 tile, 64x64x32 warps, a 4-stage cp.async pipeline.
using T = cutlass::half_t;
using Gemm = cutlass::gemm::device::Gemm<
    T, cutlass::layout::RowMajor, T, cutlass::layout::RowMajor, T, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<T, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

int gemm(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream) {
  Gemm op;
  Gemm::Arguments args({m, n, k}, {(const T*)a, k}, {(const T*)b, n}, {(T*)c, n}, {(T*)c, n},
                       {1.0f, 0.0f});
  return int(op(args, nullptr, stream));
}
"""

CPP_SRC = r"""
int gemm(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream);

torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.scalar_type() == torch::kHalf && b.scalar_type() == torch::kHalf, "fp16 inputs");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous());
  int m = a.size(0), k = a.size(1), n = b.size(1);
  TORCH_CHECK(k % 8 == 0 && n % 8 == 0, "K and N must be multiples of 8");
  auto c = torch::empty({m, n}, a.options());
  int status = gemm(a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k,
                    at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == 0, "GEMM status ", status);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return ext.forward(A.contiguous(), B.contiguous())
