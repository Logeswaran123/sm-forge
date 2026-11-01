import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void __launch_bounds__(256) gelu_kernel(const T* __restrict__ x, T* __restrict__ y,
                                                   long n) {
  long stride = (long)gridDim.x * blockDim.x;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride) {
    float v = float(x[i]);
    y[i] = T(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
  }
}

template <typename T>
static void launch(const void* x, void* y, long n, cudaStream_t stream) {
  long blocks = (n + 255) / 256;
  if (blocks > 84L * 32) blocks = 84L * 32;
  gelu_kernel<T><<<(unsigned)blocks, 256, 0, stream>>>((const T*)x, (T*)y, n);
}

void gelu(int dtype, const void* x, void* y, long n, cudaStream_t stream) {
  if (n <= 0) return;
  if (dtype == 1) launch<__half>(x, y, n, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, y, n, stream);
  else launch<float>(x, y, n, stream);
}
"""

CPP_SRC = r"""
void gelu(int dtype, const void* x, void* y, long n, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_contiguous());
  auto type = x.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  auto y = torch::empty_like(x);
  gelu(dtype, x.data_ptr(), y.data_ptr(), x.numel(), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ext.forward(x.contiguous())
