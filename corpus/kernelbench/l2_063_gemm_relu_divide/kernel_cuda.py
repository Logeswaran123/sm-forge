import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BM 64
#define BN 64
#define BK 16

// y[m, n] = relu(sum_k x[m,k] * w[n,k] + b[n]) * inv_div
template <typename T>
__global__ void __launch_bounds__(256) linear_relu_div_kernel(
    const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ b,
    T* __restrict__ y, int M, int N, int K, float inv_div) {
  __shared__ float As[BK][BM + 1];
  __shared__ float Bs[BK][BN + 1];

  int tid = threadIdx.x;
  int tx = tid % 16;
  int ty = tid / 16;
  int m0 = blockIdx.y * BM;
  int n0 = blockIdx.x * BN;

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // load tiles: BM*BK = 1024 elements, 4 per thread
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      int idx = tid + l * 256;
      int r = idx / BK;   // row within tile
      int c = idx % BK;   // k within tile
      int gm = m0 + r;
      int gk = k0 + c;
      float va = 0.f;
      if (gm < M && gk < K) va = float(x[(long)gm * K + gk]);
      As[c][r] = va;
      int gn = n0 + r;
      float vb = 0.f;
      if (gn < N && gk < K) vb = float(w[(long)gn * K + gk]);
      Bs[c][r] = vb;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a[4], bb[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < 4; ++j) bb[j] = Bs[kk][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * bb[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int gm = m0 + ty + 16 * i;
    if (gm >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int gn = n0 + tx + 16 * j;
      if (gn >= N) continue;
      float v = acc[i][j] + float(b[gn]);
      v = v > 0.f ? v : 0.f;
      y[(long)gm * N + gn] = T(v * inv_div);
    }
  }
}

template <typename T>
static void launch(const void* x, const void* w, const void* b, void* y,
                   int M, int N, int K, float inv_div, cudaStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  linear_relu_div_kernel<T><<<grid, 256, 0, stream>>>(
      (const T*)x, (const T*)w, (const T*)b, (T*)y, M, N, K, inv_div);
}

void linear_relu_div(int dtype, const void* x, const void* w, const void* b, void* y,
                     int M, int N, int K, float inv_div, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, w, b, y, M, N, K, inv_div, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, w, b, y, M, N, K, inv_div, stream);
  else launch<float>(x, w, b, y, M, N, K, inv_div, stream);
}
"""

CPP_SRC = r"""
void linear_relu_div(int dtype, const void* x, const void* w, const void* b, void* y,
                     int M, int N, int K, float inv_div, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, double divisor) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda());
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && b.is_contiguous());
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && b.dim() == 1);
  TORCH_CHECK(x.scalar_type() == w.scalar_type() && x.scalar_type() == b.scalar_type());
  int M = x.size(0);
  int K = x.size(1);
  int N = w.size(0);
  TORCH_CHECK(w.size(1) == K && b.size(0) == N);
  auto type = x.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  auto y = torch::empty({M, N}, x.options());
  linear_relu_div(dtype, x.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(),
                  M, N, K, (float)(1.0 / divisor), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).contiguous()
        w = self.linear.weight.to(x2.dtype).contiguous()
        b = self.linear.bias.to(x2.dtype).contiguous()
        y = ext.forward(x2, w, b, float(self.divisor))
        return y.view(*orig_shape[:-1], w.shape[0])
