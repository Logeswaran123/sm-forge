import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define SCAN_THREADS 256
#define SCAN_PER_THREAD 4
#define SCAN_TILE (SCAN_THREADS * SCAN_PER_THREAD)

template <typename T>
__global__ void __launch_bounds__(SCAN_THREADS) scan_kernel(const T* __restrict__ x,
                                                            T* __restrict__ y,
                                                            long L, long inner, long rows) {
  long r = blockIdx.x;
  if (r >= rows) return;
  long o = r / inner;
  long in = r % inner;
  long base = o * L * inner + in;

  __shared__ float warp_sums[SCAN_THREADS / 32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = SCAN_THREADS / 32;

  float carry = 0.f;
  for (long t0 = 0; t0 < L; t0 += SCAN_TILE) {
    float v[SCAN_PER_THREAD];
#pragma unroll
    for (int k = 0; k < SCAN_PER_THREAD; ++k) {
      long i = t0 + (long)tid * SCAN_PER_THREAD + k;
      v[k] = (i < L) ? float(x[base + i * inner]) : 0.f;
    }
#pragma unroll
    for (int k = 1; k < SCAN_PER_THREAD; ++k) v[k] += v[k - 1];
    float tot = v[SCAN_PER_THREAD - 1];

    // inclusive warp scan of thread totals
    float s = tot;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      float n = __shfl_up_sync(0xffffffffu, s, off);
      if (lane >= off) s += n;
    }
    if (lane == 31) warp_sums[warp] = s;
    __syncthreads();
    if (warp == 0) {
      float w = (lane < nwarps) ? warp_sums[lane] : 0.f;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        float n = __shfl_up_sync(0xffffffffu, w, off);
        if (lane >= off) w += n;
      }
      if (lane < nwarps) warp_sums[lane] = w;
    }
    __syncthreads();

    float prefix = carry + (warp > 0 ? warp_sums[warp - 1] : 0.f) + (s - tot);
#pragma unroll
    for (int k = 0; k < SCAN_PER_THREAD; ++k) {
      long i = t0 + (long)tid * SCAN_PER_THREAD + k;
      if (i < L) y[base + i * inner] = T(v[k] + prefix);
    }
    carry += warp_sums[nwarps - 1];
    __syncthreads();
  }
}

template <typename T>
static void launch(const void* x, void* y, long L, long inner, long rows, cudaStream_t stream) {
  if (rows <= 0 || L <= 0) return;
  scan_kernel<T><<<(unsigned)rows, SCAN_THREADS, 0, stream>>>((const T*)x, (T*)y, L, inner, rows);
}

void cumsum_launch(int dtype, const void* x, void* y, long outer, long L, long inner,
                   cudaStream_t stream) {
  long rows = outer * inner;
  if (dtype == 1) launch<__half>(x, y, L, inner, rows, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, y, L, inner, rows, stream);
  else launch<float>(x, y, L, inner, rows, stream);
}
"""

CPP_SRC = r"""
void cumsum_launch(int dtype, const void* x, void* y, long outer, long L, long inner,
                   cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, int64_t dim) {
  TORCH_CHECK(x.is_cuda());
  x = x.contiguous();
  int64_t nd = x.dim();
  if (nd == 0) {
    auto y0 = torch::empty_like(x);
    auto type0 = x.scalar_type();
    int dt0 = type0 == torch::kHalf ? 1 : type0 == torch::kBFloat16 ? 2 : 0;
    cumsum_launch(dt0, x.data_ptr(), y0.data_ptr(), 1, 1, 1, at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y0;
  }
  if (dim < 0) dim += nd;
  TORCH_CHECK(dim >= 0 && dim < nd);
  auto type = x.scalar_type();
  TORCH_CHECK(type == torch::kFloat || type == torch::kHalf || type == torch::kBFloat16);
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  long outer = 1, inner = 1;
  for (int64_t i = 0; i < dim; ++i) outer *= x.size(i);
  for (int64_t i = dim + 1; i < nd; ++i) inner *= x.size(i);
  long L = x.size(dim);
  auto y = torch::empty_like(x);
  if (x.numel() == 0) return y;
  cumsum_launch(dtype, x.data_ptr(), y.data_ptr(), outer, L, inner,
                at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        return ext.forward(x.contiguous(), int(self.dim))
