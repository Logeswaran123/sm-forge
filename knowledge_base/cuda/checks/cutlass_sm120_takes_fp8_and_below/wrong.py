import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

// CUTLASS 3.x with the SM120 collective builder and fp16 operands.
using namespace cute;
using T = cutlass::half_t;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, T, cutlass::layout::RowMajor, 8, T,
    cutlass::layout::ColumnMajor, 8, float, Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

int gemm(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream) {
  return sizeof(Mainloop) > 0 ? 1 : 1;
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
