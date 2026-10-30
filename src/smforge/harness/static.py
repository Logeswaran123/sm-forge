"""Static checks on a Triton kernel file. Source analysis only, no GPU.

Two checks reject patterns that pass an output comparison but are unsafe: a ``tl.dot`` that
accumulates in fp16 or bf16 (partial sums can overflow or lose precision on real data), and a
``while`` loop in a kernel that polls an atomic or volatile load (a spin wait between programs
deadlocks when the programs it waits for are not resident). A third rejects state kept across
calls: ``forward`` and its helpers may not assign to ``self``, so nothing derived from the weights
or the outputs is reused by a later call, as the eager and compiled baselines reuse nothing.
"""

from __future__ import annotations

import ast
import re

from smforge.harness.source import KernelSource
from smforge.models import Severity, ValidationIssue

MAX_WARPS = 32
MIN_DOT_DIM = 16
DEFAULT_NUM_STAGES = 3

_DOT_DIM = re.compile(r"^BLOCK_(?:SIZE_)?([MNK])$")
_HOST_SYNC_METHODS = frozenset({"item", "cpu", "tolist", "numpy"})
_HOST_SYNC_CALLS = frozenset({"torch.cuda.synchronize", "torch.cuda.current_stream.synchronize"})
_DOT_CALLS = frozenset({"triton.language.dot", "triton.language.dot_scaled"})
_LOW_PRECISION = frozenset({"triton.language.float16", "triton.language.bfloat16"})
_ATOMICS = re.compile(r"^triton\.language\.atomic_")


def validate_source(
    code: str,
    *,
    smem_limit: int | None = None,
    elem_bytes: int | None = None,
) -> list[ValidationIssue]:
    """Check structure, launch parameters, block shapes and host syncs.

    ``smem_limit`` (bytes per block) and ``elem_bytes`` enable the shared-memory estimate for
    ``triton.Config`` entries.
    """
    try:
        src = KernelSource(code)
    except SyntaxError as e:
        return [_error("syntax_error", f"Python syntax error: {e.msg}", e.lineno)]
    issues = _structure(src)
    issues += _constants(src)
    issues += _aranges(src)
    if smem_limit and elem_bytes:
        issues += _shared_memory(src, smem_limit, elem_bytes)
    issues += _host_syncs(src)
    issues += _unsafe_kernels(src)
    issues += _forward_state(src)
    return issues


def _forward_state(src: KernelSource) -> list[ValidationIssue]:
    issues = []
    for fn in src.forward_path():
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign | ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setattr":
                targets = [ast.Attribute(value=arg, attr="?") for arg in node.args[:1]]
            for target in targets:
                base = target.value if isinstance(target, ast.Subscript) else target
                if isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name) and base.id == "self" and target is not base:
                    issues.append(
                        _error(
                            "state_in_forward",
                            f"{fn.name} stores state on self; recompute it on every call",
                            node.lineno,
                        )
                    )
    return issues


def _unsafe_kernels(src: KernelSource) -> list[ValidationIssue]:
    issues = []
    for fn in src.functions.values():
        for call, path in src.calls(fn):
            if path not in _DOT_CALLS:
                continue
            out_dtype = next((k.value for k in call.keywords if k.arg == "out_dtype"), None)
            if out_dtype is not None and src.dotted(out_dtype) in _LOW_PRECISION:
                issues.append(
                    _error(
                        "low_precision_accumulate",
                        "tl.dot accumulates in fp16/bf16; use a float32 accumulator",
                        call.lineno,
                    )
                )
        for loop in (n for n in ast.walk(fn) if isinstance(n, ast.While)):
            polls = [
                path
                for call, path in src.calls(loop)
                if path
                and (
                    _ATOMICS.match(path)
                    or (
                        path == "triton.language.load"
                        and any(k.arg == "volatile" for k in call.keywords)
                    )
                )
            ]
            if polls:
                issues.append(
                    _error(
                        "spin_wait",
                        "a while loop polls an atomic or volatile load; waiting on other "
                        "programs deadlocks when they are not resident",
                        loop.lineno,
                    )
                )
    return issues


def _structure(src: KernelSource) -> list[ValidationIssue]:
    issues = []
    if not any(path.startswith("triton") for path in src.aliases.values()):
        issues.append(_error("missing_triton_import", "the file does not import triton"))
    if not src.triton_functions:
        issues.append(_error("missing_triton_kernel", "no @triton.jit function"))
    if src.model is None:
        issues.append(_error("missing_model", "no class Model"))
    elif "forward" not in src.methods:
        issues.append(_error("missing_forward", "Model has no forward method", src.model.lineno))
    return issues


def _constants(src: KernelSource) -> list[ValidationIssue]:
    uses_dot = any(path in _DOT_CALLS for _, path in src.calls(src.tree))
    issues = []
    for name, value, line in _int_constants(src.tree):
        if name == "num_warps" and not (_pow2(value) and value <= MAX_WARPS):
            issues.append(
                _error(
                    "num_warps",
                    f"num_warps={value} must be a power of two up to {MAX_WARPS}",
                    line,
                )
            )
        elif name == "num_stages" and value < 1:
            issues.append(_error("num_stages", f"num_stages={value} must be at least 1", line))
        elif name.startswith("BLOCK") and not _pow2(value):
            issues.append(_error("block_pow2", f"{name}={value} is not a power of two", line))
        elif uses_dot and _DOT_DIM.match(name) and value < MIN_DOT_DIM:
            issues.append(
                _error("dot_min_dim", f"{name}={value}: tl.dot needs at least {MIN_DOT_DIM}", line)
            )
    return issues


def _aranges(src: KernelSource) -> list[ValidationIssue]:
    issues = []
    for call, path in src.calls(src.tree):
        if path != "triton.language.arange" or len(call.args) != 2:
            continue
        start, end = (_int(a) for a in call.args)
        if start is not None and end is not None and not _pow2(end - start):
            issues.append(
                _error(
                    "arange_pow2",
                    f"tl.arange({start}, {end}) has length {end - start}, not a power of two",
                    call.lineno,
                )
            )
    return issues


def _shared_memory(src: KernelSource, limit: int, elem_bytes: int) -> list[ValidationIssue]:
    issues = []
    for call, path in src.calls(src.tree):
        if path != "triton.Config" or not call.args or not isinstance(call.args[0], ast.Dict):
            continue
        dims = {}
        for key, value in zip(call.args[0].keys, call.args[0].values, strict=True):
            match = _DOT_DIM.match(key.value) if isinstance(key, ast.Constant) else None
            if match and _int(value) is not None:
                dims[match.group(1)] = _int(value)
        if len(dims) != 3:
            continue
        stages = next(
            (_int(k.value) for k in call.keywords if k.arg == "num_stages"), DEFAULT_NUM_STAGES
        )
        estimate = (stages or 1) * (dims["M"] * dims["K"] + dims["K"] * dims["N"]) * elem_bytes
        if estimate > limit:
            issues.append(
                ValidationIssue(
                    check="smem_estimate",
                    severity=Severity.WARNING,
                    message=f"config needs about {estimate // 1024} KB of shared memory; "
                    f"the device allows {limit // 1024} KB per block",
                    line=call.lineno,
                )
            )
    return issues


def _host_syncs(src: KernelSource) -> list[ValidationIssue]:
    issues = []
    for fn in src.forward_path():
        for call, path in src.calls(fn):
            method = call.func.attr if isinstance(call.func, ast.Attribute) else None
            if path in _HOST_SYNC_CALLS or (path is None and method in _HOST_SYNC_METHODS):
                what = path or f".{method}()"
                issues.append(
                    ValidationIssue(
                        check="host_sync",
                        severity=Severity.WARNING,
                        message=f"{what} in the forward path waits for the GPU",
                        line=call.lineno,
                    )
                )
    return issues


def _int_constants(tree: ast.AST) -> list[tuple[str, int, int]]:
    """``(name, value, line)`` for integer constants bound to a name anywhere in the file."""
    found = []
    for node in ast.walk(tree):
        pairs: list[tuple[str, ast.expr]] = []
        if isinstance(node, ast.Call):
            pairs = [(k.arg, k.value) for k in node.keywords if k.arg]
        elif isinstance(node, ast.Dict):
            pairs = [
                (k.value, v)
                for k, v in zip(node.keys, node.values, strict=True)
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                pairs = [(target.id, node.value)]
            elif (
                isinstance(target, ast.Tuple)
                and isinstance(node.value, ast.Tuple)
                and len(target.elts) == len(node.value.elts)
            ):
                pairs = [
                    (t.id, v)
                    for t, v in zip(target.elts, node.value.elts, strict=True)
                    if isinstance(t, ast.Name)
                ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            pairs = [(node.target.id, node.value)]
        for name, value in pairs:
            if (number := _int(value)) is not None:
                found.append((name, number, value.lineno))
    return found


def _int(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _error(check: str, message: str, line: int | None = None) -> ValidationIssue:
    return ValidationIssue(check=check, severity=Severity.ERROR, message=message, line=line)
