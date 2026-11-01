import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 64x64 tile per block, 4x4 outputs per thread, scalar FFMA in float.
constexpr int TM = 64, TN = 64, TK = 16;

__global__ void __launch_bounds__(256) gemm_simt(const __half* __restrict__ A,
                                                 const __half* __restrict__ B,
                                                 __half* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TK][TM + 4];
  __shared__ float Bs[TK][TN + 4];
  int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  int row0 = blockIdx.y * TM, col0 = blockIdx.x * TN;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += TK) {
    for (int e = threadIdx.x; e < TM * TK; e += 256) {
      int m = e / TK, k = e % TK;
      As[k][m] = row0 + m < M && k0 + k < K ? __half2float(A[(long)(row0 + m) * K + k0 + k]) : 0.0f;
    }
    for (int e = threadIdx.x; e < TK * TN; e += 256) {
      int k = e / TN, n = e % TN;
      Bs[k][n] = k0 + k < K && col0 + n < N ? __half2float(B[(long)(k0 + k) * N + col0 + n]) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[k][ty * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx * 4 + j];
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
      if (r < M && c < N) C[(long)r * N + c] = __float2half(acc[i][j]);
    }
}

int gemm(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream) {
  dim3 grid((n + TN - 1) / TN, (m + TM - 1) / TM);
  gemm_simt<<<grid, 256, 0, stream>>>((const __half*)a, (const __half*)b, (__half*)c, m, n, k);
  return 0;
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
