import torch
import torch.nn as nn

from smforge.cuda import build

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void __launch_bounds__(256) conv_relu_hswish_kernel(
    const T* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    T* __restrict__ y, int N, int C, int H, int W, int OC, int K, int OH, int OW) {
  long total = (long)N * OC * OH * OW;
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int ow = (int)(idx % OW);
  long t = idx / OW;
  int oh = (int)(t % OH);
  t /= OH;
  int oc = (int)(t % OC);
  int n = (int)(t / OC);

  float acc = b[oc];
  const float* wp = w + (long)oc * C * K * K;
  for (int c = 0; c < C; ++c) {
    const T* xp = x + ((long)n * C + c) * H * W;
    for (int kh = 0; kh < K; ++kh) {
      int ih = oh + kh;
      if (ih >= H) continue;
      for (int kw = 0; kw < K; ++kw) {
        int iw = ow + kw;
        if (iw >= W) continue;
        acc += float(xp[(long)ih * W + iw]) * wp[(c * K + kh) * K + kw];
      }
    }
  }
  float r = acc > 0.f ? acc : 0.f;
  float s = (r + 3.f) / 6.f;
  s = s < 0.f ? 0.f : (s > 1.f ? 1.f : s);
  y[idx] = T(r * s);
}

template <typename T>
static void launch(const void* x, const float* w, const float* b, void* y,
                   int N, int C, int H, int W, int OC, int K, int OH, int OW,
                   cudaStream_t stream) {
  long total = (long)N * OC * OH * OW;
  if (total <= 0) return;
  long blocks = (total + 255) / 256;
  conv_relu_hswish_kernel<T><<<(unsigned)blocks, 256, 0, stream>>>(
      (const T*)x, w, b, (T*)y, N, C, H, W, OC, K, OH, OW);
}

void conv_relu_hswish(int dtype, const void* x, const float* w, const float* b, void* y,
                      int N, int C, int H, int W, int OC, int K, int OH, int OW,
                      cudaStream_t stream) {
  if (dtype == 1) launch<__half>(x, w, b, y, N, C, H, W, OC, K, OH, OW, stream);
  else if (dtype == 2) launch<__nv_bfloat16>(x, w, b, y, N, C, H, W, OC, K, OH, OW, stream);
  else launch<float>(x, w, b, y, N, C, H, W, OC, K, OH, OW, stream);
}
"""

CPP_SRC = r"""
void conv_relu_hswish(int dtype, const void* x, const float* w, const float* b, void* y,
                      int N, int C, int H, int W, int OC, int K, int OH, int OW,
                      cudaStream_t stream);

torch::Tensor forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda());
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && b.is_contiguous());
  TORCH_CHECK(w.scalar_type() == torch::kFloat && b.scalar_type() == torch::kFloat);
  TORCH_CHECK(x.dim() == 4 && w.dim() == 4);
  int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  int OC = w.size(0), K = w.size(2);
  TORCH_CHECK(w.size(1) == C && w.size(3) == K);
  int OH = H - K + 1, OW = W - K + 1;
  TORCH_CHECK(OH > 0 && OW > 0);
  auto type = x.scalar_type();
  int dtype = type == torch::kHalf ? 1 : type == torch::kBFloat16 ? 2 : 0;
  TORCH_CHECK(dtype != 0 || type == torch::kFloat);
  auto y = torch::empty({N, OC, OH, OW}, x.options());
  conv_relu_hswish(dtype, x.data_ptr(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr(),
                   N, C, H, W, OC, K, OH, OW, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.to(torch.float32).contiguous()
        b = self.conv.bias.to(torch.float32).contiguous()
        return ext.forward(x, w, b)
