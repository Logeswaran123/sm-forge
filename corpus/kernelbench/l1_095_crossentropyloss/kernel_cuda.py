import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

__device__ __forceinline__ void combine(float& m, float& s, float m2, float s2) {
  float nm = fmaxf(m, m2);
  if (nm == -INFINITY) { m = nm; s = 0.f; return; }
  s = s * __expf(m - nm) + s2 * __expf(m2 - nm);
  m = nm;
}

template <typename T>
__global__ void __launch_bounds__(256) ce_row_kernel(const T* __restrict__ x,
                                                     const long long* __restrict__ tgt,
                                                     float* __restrict__ loss,
                                                     float* __restrict__ valid,
                                                     long rows, long cols) {
  long r = blockIdx.x;
  if (r >= rows) return;
  const T* row = x + r * cols;
  float m = -INFINITY, s = 0.f;
  for (long c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = float(row[c]);
    if (v > m) { s = s * __expf(m - v) + 1.f; m = v; }
    else s += __expf(v - m);
  }
  for (int off = 16; off > 0; off >>= 1) {
    float m2 = __shfl_xor_sync(0xffffffff, m, off);
    float s2 = __shfl_xor_sync(0xffffffff, s, off);
    combine(m, s, m2, s2);
  }
  __shared__ float sm[8], ss[8];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) { sm[warp] = m; ss[warp] = s; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float M = sm[0], S = ss[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) combine(M, S, sm[w], ss[w]);
    long long t = tgt[r];
    if (t == -100 || t < 0 || t >= cols) {
      loss[r] = 0.f; valid[r] = 0.f;
    } else {
      float lse = M + logf(S);
      loss[r] = lse - float(row[t]);
      valid[r] = 1.f;
    }
  }
}

template <typename T>
__global__ void __launch_bounds__(1024) ce_mean_kernel(const float* __restrict__ loss,
                                                       const float* __restrict__ valid,
                                                       T* __restrict__ out, long rows) {
  float s = 0.f, c = 0.f;
  for (long i = threadIdx.x; i < rows; i += blockDim.x) { s += loss[i]; c += valid[i]; }
  for (int off = 16; off > 0; off >>= 1) {
    s += __shfl_xor_sync(0xffffffff, s, off);
    c += __shfl_xor_sync(0xffffffff, c, off);
  }
  __shared__ float ws[32], wc[32];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) { ws[warp] = s; wc[warp] = c; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float S = 0.f, C = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) { S += ws[w]; C += wc[w]; }
    out[0] = T(S / C);
  }
}

template <typename T>
static void launch(const void* x, const long long* t, float* loss, float* valid, void* out,
                   long rows, long cols, cudaStream_t stream) {
  if (rows > 0)
    ce_row_kernel<T><<<rows, 256, 0, stream>>>((const T*)x, t, loss, valid, rows, cols);
  ce_mean_kernel<T><<<1, 1024, 0, stream>>>(loss, valid, (T*)out, rows);
}

void cross_entropy(int dtype, const void* x, const long long* t, float* loss, float* valid,
                   void* out, long rows, long cols, cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, t, loss, valid, out, rows, cols, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, t, loss, valid, out, rows, cols, stream);
  else launch<float>(x, t, loss, valid, out, rows, cols, stream);
}
"""

CPP_SRC = r"""
void cross_entropy(int dtype, const void* x, const long long* t, float* loss, float* valid,
                   void* out, long rows, long cols, cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, torch::Tensor t) {
  TORCH_CHECK(x.is_cuda() && t.is_cuda());
  TORCH_CHECK(x.dim() == 2, "predictions must be 2D");
  TORCH_CHECK(t.scalar_type() == torch::kLong, "targets must be int64");
  x = x.contiguous();
  t = t.contiguous();
  long rows = x.size(0), cols = x.size(1);
  TORCH_CHECK(t.numel() == rows);
  auto type = x.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  TORCH_CHECK(dtype != 0 || type == torch::kFloat, "unsupported dtype");
  auto fopts = x.options().dtype(torch::kFloat);
  auto loss = torch::empty({rows > 0 ? rows : 1}, fopts);
  auto valid = torch::empty({rows > 0 ? rows : 1}, fopts);
  auto out = torch::empty({}, x.options());
  cross_entropy(dtype, x.data_ptr(), (const long long*)t.data_ptr<int64_t>(),
                loss.data_ptr<float>(), valid.data_ptr<float>(), out.data_ptr(),
                rows, cols, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return ext.forward(predictions, targets)
