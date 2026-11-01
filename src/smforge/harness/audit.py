"""Checks that a candidate computes with its own kernels, Triton or CUDA.

The source rule looks at the forward path for calls into PyTorch compute, ``nn`` layers and
dynamic imports; in a ``cuda`` file, calls on the built extension are the kernels, and the C++
binding may only allocate, reshape and check tensors. The launch rule compares the CUDA kernels
one forward launches with the baseline's: a vendor or ATen compute kernel the baseline does not
launch is an error.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from smforge.harness.source import KernelSource
from smforge.models import LaunchCategory, LaunchProfile, Severity, ValidationIssue

ALLOWED_TORCH = frozenset(
    {
        "Size",
        "arange",
        "as_strided",
        "as_tensor",
        "backends",
        "broadcast_to",
        "cat",
        "chunk",
        "cuda",
        "device",
        "empty",
        "empty_like",
        "empty_strided",
        "finfo",
        "flatten",
        "full",
        "full_like",
        "get_default_dtype",
        "iinfo",
        "inference_mode",
        "is_floating_point",
        "is_tensor",
        "narrow",
        "no_grad",
        "ones",
        "ones_like",
        "permute",
        "promote_types",
        "reshape",
        "result_type",
        "split",
        "squeeze",
        "stack",
        "tensor",
        "transpose",
        "unsqueeze",
        "zeros",
        "zeros_like",
    }
)
COMPUTE_METHODS = frozenset(
    {
        "abs",
        "addcdiv",
        "addcmul",
        "addmm",
        "addmv",
        "amax",
        "amin",
        "argmax",
        "argmin",
        "baddbmm",
        "bmm",
        "clamp",
        "clamp_max",
        "clamp_min",
        "conv1d",
        "conv2d",
        "conv3d",
        "cumsum",
        "div",
        "einsum",
        "exp",
        "exp2",
        "gelu",
        "lerp",
        "log",
        "log2",
        "log_softmax",
        "logsumexp",
        "masked_fill",
        "matmul",
        "max",
        "mean",
        "min",
        "mm",
        "mul",
        "mv",
        "norm",
        "pow",
        "prod",
        "relu",
        "rsqrt",
        "sigmoid",
        "silu",
        "softmax",
        "sort",
        "sqrt",
        "std",
        "sum",
        "tanh",
        "topk",
        "var",
        "where",
    }
)
ALLOWED_CPP = frozenset(
    {
        "empty",
        "empty_like",
        "empty_strided",
        "full",
        "full_like",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    }
)
_CPP_CALL = re.compile(r"\b(?:torch|at)::(?:nn::functional::|native::)?([a-z]\w*)\s*\(")
_CPP_METHOD = re.compile(r"\.\s*([a-z_]\w*)\s*(?:<[^<>()]*>)?\s*\(")
_DYNAMIC_CALLS = frozenset({"__import__", "compile", "eval", "exec"})
_PARAMETER_TYPES = frozenset({"Parameter", "ParameterList", "ParameterDict", "Buffer"})

_VENDOR = re.compile(
    r"gemm|gemv|cutlass|cublas|cudnn|xmma|nvjet|cutensor|winograd|fmha|flash", re.I
)
INTERFACE_FUNCTIONS = ("get_inputs", "get_init_inputs")
_ATEN_DATA = re.compile(r"copy|fill|Fill|CatArray|index|gather|scatter|transpose|contiguous")


def audit_source(code: str, allow: Iterable[str] = ()) -> list[ValidationIssue]:
    """Calls in the forward path that compute with PyTorch instead of a Triton kernel."""
    src = KernelSource(code)
    allowed = ALLOWED_TORCH | set(allow)
    issues = _dynamic_code(src) + _binding(src, allowed)
    layers = _layer_attributes(src)
    for fn in src.forward_path():
        for node in ast.walk(fn):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                issues.append(
                    _error("matmul_operator", "the @ operator runs a PyTorch matmul", node)
                )
            if isinstance(node, ast.Call):
                issues += _call(src, node, allowed, layers)
    return issues


def _call(
    src: KernelSource, call: ast.Call, allowed: frozenset[str], layers: dict[str, str]
) -> list[ValidationIssue]:
    func = call.func
    path = src.dotted(func)
    if path is not None:
        if path.startswith("torch.") and not _torch_allowed(path, allowed):
            return [_error("torch_compute_call", f"{path} computes in PyTorch", call)]
        return []
    if not isinstance(func, ast.Attribute):
        return []
    owner = func.value
    if isinstance(owner, ast.Name) and owner.id in src.extensions:
        return []
    if isinstance(owner, ast.Name) and owner.id == "self" and func.attr in layers:
        layer = layers[func.attr]
        return [_error("torch_module_call", f"self.{func.attr} is a {layer} layer", call)]
    if func.attr in COMPUTE_METHODS and func.attr not in allowed:
        return [_error("tensor_compute_method", f".{func.attr}() computes in PyTorch", call)]
    return []


def _binding(src: KernelSource, allowed: frozenset[str]) -> list[ValidationIssue]:
    """PyTorch compute called from the C++ binding of a ``cuda`` file."""
    from smforge.harness.cuda_static import code_only

    issues = []
    for source in src.cuda_sources("cpp"):
        code = code_only(source.text)
        for match in _CPP_CALL.finditer(code):
            if match.group(1) not in ALLOWED_CPP | allowed:
                issues.append(
                    ValidationIssue(
                        check="binding_compute",
                        severity=Severity.ERROR,
                        message=f"{match.group(0).rstrip('( ')} computes in PyTorch inside the binding",
                        line=source.file_line(match.start()),
                    )
                )
        for match in _CPP_METHOD.finditer(code):
            if match.group(1) in COMPUTE_METHODS - allowed:
                issues.append(
                    ValidationIssue(
                        check="binding_compute",
                        severity=Severity.ERROR,
                        message=f".{match.group(1)}() computes in PyTorch inside the binding",
                        line=source.file_line(match.start()),
                    )
                )
    return issues


def _torch_allowed(path: str, allowed: frozenset[str]) -> bool:
    rest = path.removeprefix("torch.")
    return rest in allowed or rest.split(".")[0] in allowed


def _layer_attributes(src: KernelSource) -> dict[str, str]:
    """``self.<name> = nn.<Layer>(...)`` assignments in Model methods, except parameters."""
    layers = {}
    for method in src.methods.values():
        for node in ast.walk(method):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            path = src.dotted(node.value.func) or ""
            layer = path.rsplit(".", 1)[-1]
            if not path.startswith("torch.nn.") or layer in _PARAMETER_TYPES:
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    layers[target.attr] = layer
    return layers


def _dynamic_code(src: KernelSource) -> list[ValidationIssue]:
    issues = []
    for call, path in src.calls(src.tree):
        name = call.func.id if isinstance(call.func, ast.Name) else None
        if name in _DYNAMIC_CALLS or (path or "").startswith("importlib"):
            issues.append(
                _error("dynamic_code", f"{name or path} hides what the kernel runs", call)
            )
        elif (
            name == "getattr" and call.args and (src.dotted(call.args[0]) or "").startswith("torch")
        ):
            issues.append(
                _error("dynamic_code", "getattr on torch hides what the kernel runs", call)
            )
    return issues


def audit_interface(candidate: str, baseline: str) -> list[ValidationIssue]:
    """Changes to what the harness calls: ``Model.__init__``'s arguments and the input factories."""
    new, old = KernelSource(candidate), KernelSource(baseline)
    issues = []
    if _arguments(new.methods.get("__init__")) != _arguments(old.methods.get("__init__")):
        issues.append(
            _error(
                "interface_init", "Model.__init__ takes different arguments from the stage input"
            )
        )
    for name in INTERFACE_FUNCTIONS:
        if _dump(new.functions.get(name)) != _dump(old.functions.get(name)):
            issues.append(_error("interface_inputs", f"{name} differs from the stage input"))
    return issues


def classify_launch(
    name: str, triton_names: set[str], cuda_names: set[str] = frozenset()
) -> LaunchCategory:
    if name in triton_names:
        return LaunchCategory.TRITON
    if name in cuda_names:
        return LaunchCategory.CUDA
    if name.startswith(("Memcpy", "Memset")):
        return LaunchCategory.MEMORY
    if "at::native" in name:
        if _ATEN_DATA.search(name):
            return LaunchCategory.ATEN_DATA
        return LaunchCategory.ATEN_COMPUTE
    if _VENDOR.search(name):
        return LaunchCategory.VENDOR
    return LaunchCategory.OTHER


def audit_launches(
    candidate: LaunchProfile, baseline: LaunchProfile | None = None
) -> list[ValidationIssue]:
    issues = []
    own = (LaunchCategory.TRITON, LaunchCategory.CUDA)
    if not any(launch.category in own for launch in candidate.launches):
        issues.append(_error("no_own_launch", "the forward launched none of the file's kernels"))
    known = {launch.name for launch in baseline.launches} if baseline else set()
    for launch in candidate.launches:
        if launch.name in known:
            continue
        name = short_name(launch.name)
        if launch.category is LaunchCategory.VENDOR:
            issues.append(_error("vendor_launch", f"launches library kernel {name}"))
        elif launch.category is LaunchCategory.ATEN_COMPUTE:
            issues.append(_error("aten_compute_launch", f"launches ATen compute kernel {name}"))
        elif launch.category is LaunchCategory.OTHER:
            issues.append(_error("unknown_launch", f"launches unclassified kernel {name}"))
    return issues


def _arguments(fn: ast.FunctionDef | None) -> str | None:
    return ast.dump(fn.args) if fn else None


def _dump(fn: ast.FunctionDef | None) -> str | None:
    return ast.dump(fn) if fn else None


def short_name(name: str, limit: int = 100) -> str:
    return name if len(name) <= limit else name[: limit - 3] + "..."


def _error(check: str, message: str, node: ast.AST | None = None) -> ValidationIssue:
    line = getattr(node, "lineno", None)
    return ValidationIssue(check=check, severity=Severity.ERROR, message=message, line=line)
