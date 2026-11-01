import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cstdint>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float gelu(float v) {
  return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
}

// 16 bytes per load and store: 8 halves, 8 bfloat16s or 4 floats; a scalar loop takes the tail.
template <typename T>
__global__ void __launch_bounds__(256) gelu_kernel(const T* __restrict__ x, T* __restrict__ y,
                                                   long n) {
  constexpr int V = 16 / sizeof(T);
  long vectors = n / V;
  long stride = (long)gridDim.x * blockDim.x;
  long first = blockIdx.x * (long)blockDim.x + threadIdx.x;
  for (long v = first; v < vectors; v += stride) {
    uint4 packed = reinterpret_cast<const uint4*>(x)[v];
    T* e = reinterpret_cast<T*>(&packed);
#pragma unroll
    for (int j = 0; j < V; ++j) e[j] = T(gelu(float(e[j])));
    reinterpret_cast<uint4*>(y)[v] = packed;
  }
  for (long i = vectors * V + first; i < n; i += stride) y[i] = T(gelu(float(x[i])));
}

template <typename T>
static void launch(const void* x, void* y, long n, cudaStream_t stream) {
  long blocks = (n / (16 / sizeof(T)) + 255) / 256;
  if (blocks < 1) blocks = 1;
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
