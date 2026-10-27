"""Feature probes: compile a small Triton kernel per feature and look for it in the SASS (worker side).

A probe is supported when its kernel compiles, its SASS contains the instruction the feature
should lower to, and its output matches PyTorch.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from smforge.device.kernels import (
    dot_kernel,
    dot_scaled_kernel,
    tma_device_kernel,
    tma_host_kernel,
    warp_specialize_kernel,
)
from smforge.harness.facts import capture_launches, sass_histogram
from smforge.harness.workload import cuda_healthy
from smforge.models import ProbeResult

Probe = Callable[[torch.device], tuple[torch.Tensor, torch.Tensor]]

FEATURE_OPS = frozenset({"LDGSTS", "UTMALDG", "UTMASTG"})


def _dot(dtype: torch.dtype, num_stages: int = 1, k: int = 64, kernel=dot_kernel) -> Probe:
    def run(device: torch.device):
        a = torch.randn(64, k, device=device).to(dtype)
        b = torch.randn(k, 64, device=device).to(dtype)
        c = torch.empty(64, 64, device=device)
        kernel[(1, 1)](
            a, b, c, 64, 64, k, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_stages=num_stages
        )
        return c, a.float() @ b.float()

    return run


def _tma_host(device: torch.device):
    a = torch.randn(128, 128, device=device, dtype=torch.float16)
    b = torch.randn(128, 128, device=device, dtype=torch.float16)
    c = torch.empty(128, 128, device=device)
    descriptors = [
        TensorDescriptor.from_tensor(t, block)
        for t, block in ((a, [64, 32]), (b, [32, 64]), (c, [64, 64]))
    ]
    tma_host_kernel[(2, 2)](*descriptors, 128, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return c, a.float() @ b.float()


def _tma_device(device: torch.device):
    triton.set_allocator(lambda size, _, __: torch.empty(size, dtype=torch.int8, device=device))
    a = torch.randn(128, 128, device=device, dtype=torch.float16)
    b = torch.randn(128, 128, device=device, dtype=torch.float16)
    c = torch.empty(128, 128, device=device)
    tma_device_kernel[(2, 2)](a, b, c, 128, 128, 128, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    return c, a.float() @ b.float()


def _dot_scaled(device: torch.device):
    a = torch.randn(128, 128, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(128, 128, device=device).to(torch.float8_e4m3fn)
    unit = torch.full((128, 4), 127, device=device, dtype=torch.uint8)
    c = torch.empty(128, 128, device=device)
    dot_scaled_kernel[(1,)](a, b, unit, unit, c, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128)
    return c, a.float() @ b.float()


PROBES: dict[str, tuple[Probe, str | None]] = {
    "mma_fp16": (_dot(torch.float16), "MMA"),
    "mma_bf16": (_dot(torch.bfloat16), "MMA"),
    "mma_tf32": (_dot(torch.float32), "MMA"),
    "mma_fp8": (_dot(torch.float8_e4m3fn), "MMA"),
    "cp_async": (_dot(torch.float16, num_stages=3, k=256), "LDGSTS"),
    "tma_host": (_tma_host, "UTMALDG"),
    "tma_device": (_tma_device, "UTMALDG"),
    "dot_scaled": (_dot_scaled, "MMA"),
    "warp_specialize": (_dot(torch.float16, k=256, kernel=warp_specialize_kernel), None),
}


def run_probe(name: str, device: torch.device) -> ProbeResult:
    run, expected = PROBES[name]
    with capture_launches() as records:
        try:
            out, reference = run(device)
            torch.cuda.synchronize()
        except Exception as e:
            if not cuda_healthy():
                raise
            return ProbeResult(name=name, compiled=False, error=_first_line(e))
    histogram = sass_histogram(records[-1][0])
    return ProbeResult(
        name=name,
        compiled=True,
        lowered=None if expected is None else any(expected in op for op in histogram),
        correct=bool(torch.allclose(out, reference, rtol=2e-2, atol=5e-2)),
        instructions={op: n for op, n in histogram.items() if "MMA" in op or op in FEATURE_OPS},
    )


def _first_line(error: Exception) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text.splitlines()[0] if text else ''}".strip()
