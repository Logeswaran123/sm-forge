import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define STATS_THREADS 256
#define FIN_THREADS 128
#define APPLY_THREADS 256

template <typename T>
__global__ void __launch_bounds__(STATS_THREADS) gn_stats_kernel(const T* __restrict__ x,
                                                                 double* __restrict__ part,
                                                                 long M, int P) {
  long ng = blockIdx.x;
  int p = blockIdx.y;
  long chunk = (M + P - 1) / P;
  long start = (long)p * chunk;
  long end = start + chunk;
  if (end > M) end = M;
  const T* base = x + ng * M;
  float s = 0.f, ss = 0.f;
  for (long i = start + threadIdx.x; i < end; i += STATS_THREADS) {
    float v = float(base[i]);
    s += v;
    ss += v * v;
  }
  __shared__ double sh_s[STATS_THREADS];
  __shared__ double sh_ss[STATS_THREADS];
  sh_s[threadIdx.x] = (double)s;
  sh_ss[threadIdx.x] = (double)ss;
  __syncthreads();
  for (int off = STATS_THREADS / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) {
      sh_s[threadIdx.x] += sh_s[threadIdx.x + off];
      sh_ss[threadIdx.x] += sh_ss[threadIdx.x + off];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    long idx = ng * (long)P + p;
    part[idx * 2 + 0] = sh_s[0];
    part[idx * 2 + 1] = sh_ss[0];
  }
}

__global__ void __launch_bounds__(FIN_THREADS) gn_finalize_kernel(const double* __restrict__ part,
                                                                  float* __restrict__ stats,
                                                                  long NG, int P, long M, float eps) {
  long ng = blockIdx.x * (long)FIN_THREADS + threadIdx.x;
  if (ng >= NG) return;
  double s = 0.0, ss = 0.0;
  for (int p = 0; p < P; ++p) {
    long idx = ng * (long)P + p;
    s += part[idx * 2 + 0];
    ss += part[idx * 2 + 1];
  }
  double mean = s / (double)M;
  double var = ss / (double)M - mean * mean;
  if (var < 0.0) var = 0.0;
  stats[ng * 2 + 0] = (float)mean;
  stats[ng * 2 + 1] = (float)(1.0 / sqrt(var + (double)eps));
}

template <typename T>
__global__ void __launch_bounds__(APPLY_THREADS) gn_apply_kernel(const T* __restrict__ x,
                                                                 T* __restrict__ y,
                                                                 const float* __restrict__ stats,
                                                                 const float* __restrict__ w,
                                                                 const float* __restrict__ b,
                                                                 long total, long HW, int C, long M) {
  long stride = (long)gridDim.x * APPLY_THREADS;
  for (long i = blockIdx.x * (long)APPLY_THREADS + threadIdx.x; i < total; i += stride) {
    long ng = i / M;
    int c = (int)((i / HW) % C);
    float mean = stats[ng * 2 + 0];
    float rstd = stats[ng * 2 + 1];
    float v = float(x[i]);
    y[i] = T((v - mean) * rstd * w[c] + b[c]);
  }
}

template <typename T>
static void launch_all(const void* x, void* y, const float* w, const float* b,
                       double* part, float* stats, long N, int C, int G, long HW,
                       int P, float eps, cudaStream_t stream) {
  long Cg = C / G;
  long M = Cg * HW;
  long NG = N * (long)G;
  if (NG == 0) return;
  dim3 sgrid((unsigned)NG, (unsigned)P);
  gn_stats_kernel<T><<<sgrid, STATS_THREADS, 0, stream>>>((const T*)x, part, M, P);
  gn_finalize_kernel<<<(unsigned)((NG + FIN_THREADS - 1) / FIN_THREADS), FIN_THREADS, 0, stream>>>(
      part, stats, NG, P, M, eps);
  long total = NG * M;
  if (total == 0) return;
  long blocks = (total + APPLY_THREADS - 1) / APPLY_THREADS;
  if (blocks > 84L * 64L) blocks = 84L * 64L;
  gn_apply_kernel<T><<<(unsigned)blocks, APPLY_THREADS, 0, stream>>>(
      (const T*)x, (T*)y, stats, w, b, total, HW, C, M);
}

void group_norm(int dtype, const void* x, void* y, const float* w, const float* b,
                double* part, float* stats, long N, int C, int G, long HW,
                int P, float eps, cudaStream_t stream) {
  if (dtype == 1) launch_all<__half>(x, y, w, b, part, stats, N, C, G, HW, P, eps, stream);
  else if (dtype == 2) launch_all<__nv_bfloat16>(x, y, w, b, part, stats, N, C, G, HW, P, eps, stream);
  else launch_all<float>(x, y, w, b, part, stats, N, C, G, HW, P, eps, stream);
}
"""

CPP_SRC = r"""
void group_norm(int dtype, const void* x, void* y, const float* w, const float* b,
                double* part, float* stats, long N, int C, int G, long HW,
                int P, float eps, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, int64_t G, double eps) {
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.is_contiguous());
  TORCH_CHECK(x.dim() >= 2);
  TORCH_CHECK(w.scalar_type() == torch::kFloat && b.scalar_type() == torch::kFloat);
  TORCH_CHECK(w.is_contiguous() && b.is_contiguous());
  long N = x.size(0);
  int C = (int)x.size(1);
  TORCH_CHECK(C % G == 0);
  TORCH_CHECK(w.numel() == C && b.numel() == C);
  long HW = 1;
  for (int d = 2; d < x.dim(); ++d) HW *= x.size(d);
  auto type = x.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  TORCH_CHECK(dtype != 0 || type == torch::kFloat);
  int P = 64;
  long NG = N * G;
  auto y = torch::empty_like(x);
  auto part = torch::empty({NG > 0 ? NG * P * 2 : 1}, x.options().dtype(torch::kDouble));
  auto stats = torch::empty({NG > 0 ? NG * 2 : 1}, x.options().dtype(torch::kFloat));
  group_norm(dtype, x.data_ptr(), y.data_ptr(), w.data_ptr<float>(), b.data_ptr<float>(),
             part.data_ptr<double>(), stats.data_ptr<float>(), N, C, (int)G, HW, P, (float)eps,
             at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        C = x.shape[1]
        if self.gn.weight is not None:
            w = self.gn.weight.to(torch.float32).contiguous()
            b = self.gn.bias.to(torch.float32).contiguous()
        else:
            w = torch.ones(C, device=x.device, dtype=torch.float32)
            b = torch.zeros(C, device=x.device, dtype=torch.float32)
        return ext.forward(x, w, b, int(self.gn.num_groups), float(self.gn.eps))
