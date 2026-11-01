from pathlib import Path

from smforge.harness.audit import audit_source
from smforge.harness.static import validate_source
from smforge.models import Severity

FIXTURE = (Path(__file__).parent / "kernels" / "linear_relu" / "kernel_cuda.py").read_text()

KERNEL = """
__global__ void __launch_bounds__(256) scale(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = 2.0f * x[i];
}
"""
LAUNCH = """
void run(const float* x, float* y, int n, cudaStream_t stream) {
  scale<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
}
"""
BINDING = """
void run(const float* x, float* y, int n, cudaStream_t stream);
torch::Tensor forward(torch::Tensor x) {
  auto y = torch::empty_like(x);
  run(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""


def kernel_file(cuda: str = KERNEL + LAUNCH, cpp: str = BINDING, forward: str = "ext.forward(x)"):
    return f'''import torch
from torch import nn

from smforge.cuda import build

CUDA_SRC = r"""{cuda}"""

CPP_SRC = r"""{cpp}"""

ext = build(CUDA_SRC, CPP_SRC, ["forward"])


class Model(nn.Module):
    def forward(self, x):
        return {forward}
'''


def found(code: str) -> dict[str, tuple[str, int | None]]:
    issues = validate_source(code) + audit_source(code)
    return {i.check: (i.severity, i.line) for i in issues}


def test_clean_files():
    assert found(FIXTURE) == {}
    assert found(kernel_file()) == {}


def test_errors():
    code = kernel_file(cpp=BINDING.replace("  C10_CUDA_KERNEL_LAUNCH_CHECK();\n", ""))
    assert found(code) == {"launch_check": (Severity.ERROR, 13)}

    alloc = LAUNCH.replace("  scale", "  float* tmp; cudaMalloc(&tmp, 4);\n  scale")
    assert found(kernel_file(KERNEL + alloc))["device_alloc"] == (Severity.ERROR, 13)

    state = BINDING.replace("  auto y", "  static torch::Tensor cache;\n  auto y")
    assert found(kernel_file(cpp=state))["static_state"][0] is Severity.ERROR

    header = "#include <ATen/ATen.h>\n" + KERNEL + LAUNCH
    assert found(kernel_file(header))["torch_header"] == (Severity.ERROR, 6)

    compute = BINDING.replace("auto y = torch::empty_like(x);", "auto y = torch::relu(x);")
    issues = found(kernel_file(cpp=compute))
    assert issues["binding_compute"] == (Severity.ERROR, 20)
    method = BINDING.replace("auto y = torch::empty_like(x);", "auto y = x.matmul(x);")
    assert "binding_compute" in found(kernel_file(cpp=method))

    assert "missing_build" in found(kernel_file().replace("ext = build(", "ext = dict("))
    no_kernel = kernel_file("void run(const float* x, float* y, int n, cudaStream_t s) {}")
    assert "missing_cuda_kernel" in found(no_kernel)


def test_warnings():
    loose = KERNEL.replace("__launch_bounds__(256) ", "").replace("if (i < n) ", "")
    issues = found(kernel_file(loose + LAUNCH.replace(", 0, stream>>>", ">>>")))
    assert issues == {
        "launch_bounds": (Severity.WARNING, 7),
        "bounds_check": (Severity.WARNING, 7),
        "launch_stream": (Severity.WARNING, 13),
    }
    stride = KERNEL.replace(
        "if (i < n) y[i] = 2.0f * x[i];",
        "for (; i < n; i += gridDim.x * blockDim.x) y[i] = 2.0f * x[i];",
    )
    assert found(kernel_file(stride + LAUNCH)) == {}
    sync = LAUNCH.replace("  scale", "  cudaDeviceSynchronize();\n  scale")
    assert found(kernel_file(KERNEL + sync)) == {"host_sync": (Severity.WARNING, 13)}
    # comments and strings do not count
    commented = KERNEL + LAUNCH + '// cudaMalloc(&p, 4);\nconst char* s = "cudaFree(";\n'
    assert found(kernel_file(commented)) == {}


def test_extension_calls_are_kernels():
    assert found(kernel_file(forward="ext.relu(x)")) == {}
    assert "tensor_compute_method" in found(kernel_file(forward="x.relu()"))
