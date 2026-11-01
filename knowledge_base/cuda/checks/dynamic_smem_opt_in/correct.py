import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float gelu(float v) {
  return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
}

// Each block stages 16384 floats (64 KB) of x in dynamic shared memory, then writes gelu of them.
constexpr int CHUNK = 16384;

template <typename T>
__global__ void __launch_bounds__(256) gelu_staged(const T* __restrict__ x, T* __restrict__ y,
                                                   long n) {
  extern __shared__ float stage[];
  for (long start = (long)blockIdx.x * CHUNK; start < n; start += (long)gridDim.x * CHUNK) {
    for (int i = threadIdx.x; i < CHUNK; i += 256)
      stage[i] = start + i < n ? float(x[start + i]) : 0.0f;
    __syncthreads();
    for (int i = threadIdx.x; i < CHUNK; i += 256)
      if (start + i < n) y[start + i] = T(gelu(stage[i]));
    __syncthreads();
  }
}

template <typename T>
static void launch(const void* x, void* y, long n, cudaStream_t stream) {
  long blocks = (n + CHUNK - 1) / CHUNK;
  if (blocks > 84L * 2) blocks = 84L * 2;
  cudaFuncSetAttribute(gelu_staged<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       CHUNK * sizeof(float));
  gelu_staged<T><<<(unsigned)blocks, 256, CHUNK * sizeof(float), stream>>>((const T*)x, (T*)y, n);
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
  auto y = torch::empty_like(x);
  TORCH_CHECK((uintptr_t)x.data_ptr() % 16 == 0 && (uintptr_t)y.data_ptr() % 16 == 0,
              "16-byte aligned tensors");
  auto type = x.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
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
