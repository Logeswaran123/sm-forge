import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define LN_THREADS 256

__device__ __forceinline__ float block_reduce_sum(float v, float* sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  __syncthreads();  // protect sh reuse across calls
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  int nw = blockDim.x >> 5;
  float r = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.f;
  if (warp == 0) {
    for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    if (lane == 0) sh[0] = r;
  }
  __syncthreads();
  return sh[0];
}

template <typename T>
__global__ void __launch_bounds__(LN_THREADS) layernorm_kernel(
    const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    T* __restrict__ y, long rows, long n, float eps) {
  __shared__ float sh[32];
  long row = blockIdx.x;
  if (row >= rows) return;
  const T* xr = x + row * n;
  T* yr = y + row * n;

  float s = 0.f;
  for (long i = threadIdx.x; i < n; i += blockDim.x) s += float(xr[i]);
  float mean = block_reduce_sum(s, sh) / (float)n;

  float v = 0.f;
  for (long i = threadIdx.x; i < n; i += blockDim.x) {
    float d = float(xr[i]) - mean;
    v += d * d;
  }
  float var = block_reduce_sum(v, sh) / (float)n;
  float rstd = rsqrtf(var + eps);

  for (long i = threadIdx.x; i < n; i += blockDim.x) {
    float val = (float(xr[i]) - mean) * rstd * w[i] + b[i];
    yr[i] = T(val);
  }
}

template <typename T>
static void launch(const void* x, const float* w, const float* b, void* y, long rows, long n,
                   float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  layernorm_kernel<T><<<(unsigned)rows, LN_THREADS, 0, stream>>>(
      (const T*)x, w, b, (T*)y, rows, n, eps);
}

void layernorm(int dtype, const void* x, const float* w, const float* b, void* y, long rows,
               long n, float eps, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, w, b, y, rows, n, eps, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, w, b, y, rows, n, eps, stream);
  else launch<float>(x, w, b, y, rows, n, eps, stream);
}
"""

CPP_SRC = r"""
void layernorm(int dtype, const void* x, const float* w, const float* b, void* y, long rows,
               long n, float eps, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous());
  TORCH_CHECK(w.is_contiguous() && b.is_contiguous());
  TORCH_CHECK(w.scalar_type() == torch::kFloat && b.scalar_type() == torch::kFloat);
  auto type = x.scalar_type();
  TORCH_CHECK(type == torch::kHalf || type == torch::kBFloat16 || type == torch::kFloat);
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  long n = x.size(-1);
  TORCH_CHECK(w.numel() == n && b.numel() == n);
  long rows = n > 0 ? x.numel() / n : 0;
  auto y = torch::empty_like(x);
  if (rows > 0 && n > 0) {
    layernorm(dtype, x.data_ptr(), (const float*)w.data_ptr(), (const float*)b.data_ptr(),
              y.data_ptr(), rows, n, (float)eps, at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(hidden_size))
        self.bias = nn.Parameter(0.1 * torch.randn(hidden_size))
        self.eps = eps

    def forward(self, x):
        x = x.contiguous()
        w = self.weight.to(torch.float32).contiguous()
        b = self.bias.to(torch.float32).contiguous()
        return ext.forward(x, w, b, float(self.eps))
