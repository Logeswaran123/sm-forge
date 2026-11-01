import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// relu(x @ W.T + b) / divisor with a 64x64 tile per block and 4x4 outputs per thread. The tiles
// of x [M, K] and W [N, K] are stored transposed, as [k][m] and [k][n], so the 16 threads that store one row's k values hit the same bank.
constexpr int TM = 64, TN = 64, TK = 16, PAD = 0;

__global__ void __launch_bounds__(256) linear_relu_div(const __half* __restrict__ X,
                                                       const __half* __restrict__ W,
                                                       const __half* __restrict__ bias,
                                                       __half* __restrict__ Y, int M, int N,
                                                       int K, float divisor) {
  __shared__ float Xs[TK][TM + PAD];
  __shared__ float Ws[TK][TN + PAD];
  int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  int row0 = blockIdx.y * TM, col0 = blockIdx.x * TN;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += TK) {
    for (int e = threadIdx.x; e < TM * TK; e += 256) {
      int m = e / TK, k = e % TK;
      Xs[k][m] = row0 + m < M && k0 + k < K ? __half2float(X[(long)(row0 + m) * K + k0 + k]) : 0.0f;
    }
    for (int e = threadIdx.x; e < TN * TK; e += 256) {
      int n = e / TK, k = e % TK;
      Ws[k][n] = col0 + n < N && k0 + k < K ? __half2float(W[(long)(col0 + n) * K + k0 + k]) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = Xs[k][ty * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Ws[k][tx * 4 + j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      int r = row0 + ty * 4 + i, c = col0 + tx * 4 + j;
      if (r < M && c < N)
        Y[(long)r * N + c] = __float2half(fmaxf(acc[i][j] + __half2float(bias[c]), 0.0f) / divisor);
    }
}

void linear(const void* x, const void* w, const void* b, void* y, int m, int n, int k,
            float divisor, cudaStream_t stream) {
  dim3 grid((n + TN - 1) / TN, (m + TM - 1) / TM);
  linear_relu_div<<<grid, 256, 0, stream>>>((const __half*)x, (const __half*)w, (const __half*)b,
                                            (__half*)y, m, n, k, divisor);
}
"""

CPP_SRC = r"""
void linear(const void* x, const void* w, const void* b, void* y, int m, int n, int k,
            float divisor, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, double divisor) {
  TORCH_CHECK(x.scalar_type() == torch::kHalf && w.scalar_type() == torch::kHalf, "fp16 inputs");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && b.is_contiguous());
  int m = x.size(0), k = x.size(1), n = w.size(0);
  auto y = torch::empty({m, n}, x.options());
  linear(x.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(), m, n, k, divisor,
         at::cuda::getCurrentCUDAStream());
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
        return ext.forward(x.contiguous(), self.linear.weight, self.linear.bias, self.divisor)
