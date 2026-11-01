import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

#define THREADS 1024

__device__ __forceinline__ void combine(float& m, float& s, float m2, float s2) {
  float mn = fmaxf(m, m2);
  float a = (m == -INFINITY) ? 0.f : s * expf(m - mn);
  float b = (m2 == -INFINITY) ? 0.f : s2 * expf(m2 - mn);
  m = mn;
  s = a + b;
}

template <typename T>
__global__ void __launch_bounds__(THREADS) softmax_kernel(const T* __restrict__ x,
                                                          float* __restrict__ y,
                                                          long rows, long cols) {
  long row = blockIdx.x;
  if (row >= rows) return;
  const T* xr = x + row * cols;
  float* yr = y + row * cols;

  float m = -INFINITY, s = 0.f;
  for (long i = threadIdx.x; i < cols; i += THREADS) {
    float v = float(xr[i]);
    if (v > m) {
      s = (m == -INFINITY) ? 0.f : s * expf(m - v);
      m = v;
    }
    if (v != -INFINITY) s += expf(v - m);
  }

  // warp reduce
  for (int off = 16; off > 0; off >>= 1) {
    float m2 = __shfl_xor_sync(0xffffffff, m, off);
    float s2 = __shfl_xor_sync(0xffffffff, s, off);
    combine(m, s, m2, s2);
  }

  __shared__ float sm[THREADS / 32];
  __shared__ float ss[THREADS / 32];
  __shared__ float fm, fs;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) { sm[warp] = m; ss[warp] = s; }
  __syncthreads();
  if (warp == 0) {
    const int nw = THREADS / 32;
    m = (lane < nw) ? sm[lane] : -INFINITY;
    s = (lane < nw) ? ss[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) {
      float m2 = __shfl_xor_sync(0xffffffff, m, off);
      float s2 = __shfl_xor_sync(0xffffffff, s, off);
      combine(m, s, m2, s2);
    }
    if (lane == 0) { fm = m; fs = s; }
  }
  __syncthreads();
  float M = fm;
  float inv = 1.f / fs;

  for (long i = threadIdx.x; i < cols; i += THREADS) {
    float v = float(xr[i]);
    yr[i] = expf(v - M) * inv;
  }
}

template <typename T>
static void launch(const void* x, void* y, long rows, long cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;
  softmax_kernel<T><<<(unsigned)rows, THREADS, 0, stream>>>((const T*)x, (float*)y, rows, cols);
}

void softmax_rows(int dtype, const void* x, void* y, long rows, long cols, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, y, rows, cols, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, y, rows, cols, stream);
  else launch<float>(x, y, rows, cols, stream);
}
"""

CPP_SRC = r"""
void softmax_rows(int dtype, const void* x, void* y, long rows, long cols, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.is_contiguous());
  TORCH_CHECK(x.dim() >= 1);
  auto type = x.scalar_type();
  TORCH_CHECK(type == torch::kFloat || type == torch::kHalf || type == torch::kBFloat16);
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  long cols = x.size(-1);
  long rows = cols > 0 ? x.numel() / cols : 0;
  auto y = torch::empty(x.sizes(), x.options().dtype(torch::kFloat));
  softmax_rows(dtype, x.data_ptr(), y.data_ptr(), rows, cols, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return ext.forward(logits.contiguous())
