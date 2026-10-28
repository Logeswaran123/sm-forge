import pytest

from smforge.config import CorrectnessConfig
from smforge.harness.audit import audit_interface, audit_launches, audit_source, classify_launch
from smforge.harness.tolerance import resolve_tolerance
from smforge.models import Launch, LaunchCategory, LaunchProfile
from smforge.spec import Variant

HEADER = """
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu
import triton
import triton.language as tl


@triton.jit
def kernel(x):
    tl.sum(tl.load(x), axis=0)

"""


def model(body: str, init: str = "        pass") -> str:
    return (
        HEADER
        + "class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
        + init
        + "\n\n    def forward(self, x):\n"
        + body
    )


def checks(issues):
    return sorted(issue.check for issue in issues)


def test_layout_ops_pass():
    code = model(
        "        y = torch.empty_like(x).contiguous().view(-1)\n"
        "        kernel[(1,)](x.to(torch.float16).t().contiguous())\n"
        "        return torch.cat([y, y]).reshape(2, -1)\n"
    )
    assert audit_source(code) == []


@pytest.mark.parametrize(
    ("body", "check"),
    [
        ("        return torch.matmul(x, x)\n", "torch_compute_call"),
        ("        return F.linear(x, x)\n", "torch_compute_call"),
        ("        return relu(x)\n", "torch_compute_call"),
        ("        return torch.ops.aten.mm(x, x)\n", "torch_compute_call"),
        ("        return x @ x\n", "matmul_operator"),
        ("        return x.sum(dim=0)\n", "tensor_compute_method"),
        ("        return getattr(torch, 'mm')(x, x)\n", "dynamic_code"),
        ("        return importlib.import_module('torch').mm(x, x)\n", "dynamic_code"),
    ],
)
def test_forward_compute_is_flagged(body, check):
    assert check in checks(audit_source(model(body)))


def test_helpers_reached_from_forward_are_checked():
    code = (
        model("        return self._post(x)\n")
        + "\n    def _post(self, y):\n        return y.softmax(-1)\n"
    )
    assert checks(audit_source(code)) == ["tensor_compute_method"]


def test_layer_called_in_forward():
    code = model(
        "        return self.fc(x)\n",
        init="        self.fc = nn.Linear(4, 4)\n        self.w = nn.Parameter(torch.ones(4))",
    )
    assert checks(audit_source(code)) == ["torch_module_call"]


def test_init_is_not_checked():
    code = model("        return x\n", init="        self.scale = torch.ones(4).sum()")
    assert audit_source(code) == []


def test_allow_list():
    assert audit_source(model("        return x.sum()\n"), allow=["sum"]) == []


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("matmul_kernel", LaunchCategory.TRITON),
        ("Memset (Device)", LaunchCategory.MEMORY),
        (
            "void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16>(",
            LaunchCategory.VENDOR,
        ),
        ("sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize", LaunchCategory.VENDOR),
        ("nvjet_tst_128x64_64x6_1x1_v_bz_coopA_TNT", LaunchCategory.VENDOR),
        (
            "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<c10::Half>>",
            LaunchCategory.ATEN_COMPUTE,
        ),
        (
            "void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda>",
            LaunchCategory.ATEN_DATA,
        ),
        (
            "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>>",
            LaunchCategory.ATEN_DATA,
        ),
        ("some_custom_kernel", LaunchCategory.OTHER),
    ],
)
def test_classify_launch(name, category):
    assert classify_launch(name, {"matmul_kernel"}) is category


def profile(*launches):
    return LaunchProfile(
        launches=[Launch(name=n, category=c, calls=1, duration_us=1.0) for n, c in launches],
        kernel_us=float(len(launches)),
    )


def test_audit_launches():
    gemm = ("cutlass_gemm", LaunchCategory.VENDOR)
    triton = ("matmul_kernel", LaunchCategory.TRITON)
    reduce = ("at::native::reduce_kernel", LaunchCategory.ATEN_COMPUTE)
    assert audit_launches(profile(triton)) == []
    assert checks(audit_launches(profile(gemm))) == ["no_triton_launch", "vendor_launch"]
    assert checks(audit_launches(profile(triton, gemm, reduce), profile(gemm))) == [
        "aten_compute_launch"
    ]
    assert checks(audit_launches(profile(triton, ("helper", LaunchCategory.OTHER)))) == [
        "unknown_launch"
    ]


def test_tolerance_precedence():
    variant = Variant(params=[], rtol=1e-3)
    config = CorrectnessConfig(rtol=5e-2, atol=5e-2)
    assert resolve_tolerance("float16") == (1e-2, 1e-2)
    assert resolve_tolerance("float16", variant, config) == (1e-3, 5e-2)
    assert resolve_tolerance("float16", variant, config, rtol=0.5, atol=0.0) == (0.5, 0.0)
    assert resolve_tolerance("int32") == (0.0, 0.0)


INTERFACE = """
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

    def forward(self, x):
        return x


def get_inputs():
    return [torch.randn(8, 16)]


def get_init_inputs():
    return [16, 32]
"""


def test_audit_interface():
    assert audit_interface(INTERFACE + "\n# tuned\n", INTERFACE) == []
    renamed = INTERFACE.replace("in_features, out_features", "in_features, out_features, bias=True")
    assert checks(audit_interface(renamed, INTERFACE)) == ["interface_init"]
    shapes = INTERFACE.replace("torch.randn(8, 16)", "torch.randn(4, 16)")
    assert checks(audit_interface(shapes, INTERFACE)) == ["interface_inputs"]
    dropped = INTERFACE[: INTERFACE.index("def get_init_inputs")]
    assert checks(audit_interface(dropped, INTERFACE)) == ["interface_inputs"]
