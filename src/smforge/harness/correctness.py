"""Comparing a candidate's outputs with the oracle's (worker side)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn

from smforge.harness.launches import triton_names
from smforge.harness.workload import Workload, clone, load_module
from smforge.models import CheckResult, OutputDiff


def flatten(value: Any) -> list:
    if isinstance(value, list | tuple):
        return [item for v in value for item in flatten(v)]
    return [value]


def compare_outputs(
    expected: Any, actual: Any, rtol: float, atol: float
) -> tuple[list[OutputDiff], str | None]:
    """Per-output differences, or an error when the outputs do not line up at all."""
    expected, actual = flatten(expected), flatten(actual)
    if len(expected) != len(actual):
        return [], f"expected {len(expected)} outputs, got {len(actual)}"
    diffs = []
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
        if not isinstance(e, torch.Tensor):
            if isinstance(a, torch.Tensor) or e != a:
                return diffs, f"output {i}: expected {e!r}, got {a!r}"
            continue
        if not isinstance(a, torch.Tensor):
            return diffs, f"output {i}: expected a tensor, got {type(a).__name__}"
        if e.shape != a.shape:
            return diffs, f"output {i}: expected shape {list(e.shape)}, got {list(a.shape)}"
        diffs.append(_diff(i, e, a, rtol, atol))
    return diffs, None


def _diff(index: int, expected: torch.Tensor, actual: torch.Tensor, rtol, atol) -> OutputDiff:
    e = expected.detach().float()
    a = actual.detach().to(e.device).float()
    bad = ~torch.isclose(a, e, rtol=rtol, atol=atol, equal_nan=True)
    err = (a - e).abs()
    finite = torch.isfinite(err)
    significant = finite & (e.abs() > atol)
    max_abs = float(err[finite].max()) if finite.any() else 0.0
    max_rel = float((err[significant] / e[significant].abs()).max()) if significant.any() else 0.0
    mismatched = int(bad.sum())
    first_bad = None
    if mismatched:
        flat = int(bad.flatten().nonzero()[0])
        first_bad = [int(i) for i in np.unravel_index(flat, tuple(e.shape))]
    return OutputDiff(
        index=index,
        shape=list(e.shape),
        total=e.numel(),
        mismatched=mismatched,
        max_abs=max_abs,
        max_rel=max_rel,
        nan=int((a.isnan() & ~e.isnan()).sum()),
        inf=int((a.isinf() & ~e.isinf()).sum()),
        first_bad=first_bad,
    )


def copy_weights(source: nn.Module, target: nn.Module) -> str:
    """Give ``target`` the oracle's weights: by name, else by position when shapes line up."""
    state = source.state_dict()
    if not state:
        return "none"
    try:
        target.load_state_dict(state)
        return "by_name"
    except RuntimeError:
        pass
    sources, targets = list(state.values()), list(target.state_dict().values())
    if len(sources) != len(targets) or any(
        s.shape != t.shape for s, t in zip(sources, targets, strict=True)
    ):
        return "seed"
    with torch.no_grad():
        for s, t in zip(sources, targets, strict=True):
            t.copy_(s)
    return "by_position"


class ReferenceCache:
    """Oracle outputs on disk, keyed by oracle source, workload and input draw."""

    def __init__(self, root: Path, device: torch.device):
        self.root = root
        self.device = device

    def get(self, key: str, compute: Callable[[], Any]) -> Any:
        path = self.root / f"{key}.pt"
        if path.exists():
            return _move(torch.load(path), self.device)
        value = compute()
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(_move(value, torch.device("cpu")), tmp)
        tmp.replace(path)
        return value


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list | tuple):
        return type(value)(_move(v, device) for v in value)
    return value


def _cast(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(dtype)
    if isinstance(value, list | tuple):
        return type(value)(_cast(v, dtype) for v in value)
    return value


def oracle_dtype(workload: Workload, module: ModuleType) -> torch.dtype | None:
    """float64 when a float32 workload's oracle is plain PyTorch, else None (the workload's).

    Eager float32 is not always within float32's tolerance of the exact result: a cuDNN algorithm
    picked for a large batch, or a sum over millions of terms, can leave it further off than a
    careful kernel.
    """
    if workload.dtype == torch.float32 and not triton_names(module):
        return torch.float64
    return None


def check_kernel(
    candidate: Path,
    oracle: Path,
    workload: Workload,
    rtol: float,
    atol: float,
    draws: int,
    cache: ReferenceCache,
) -> CheckResult:
    """Run oracle and candidate on ``draws`` input draws in one process.

    The candidate model is built once, so a module that caches its first output fails the
    second draw.
    """
    oracle_module = load_module(oracle)
    upcast = oracle_dtype(workload, oracle_module)
    reference = workload.build(oracle_module, upcast)
    model = workload.build(load_module(candidate))
    weights = copy_weights(reference, model)
    digest = hashlib.sha256(oracle.read_bytes())
    digest.update(workload.key().encode())
    digest.update(f"{torch.__version__} {upcast}".encode())
    result = partial(
        CheckResult,
        rtol=rtol,
        atol=atol,
        weights=weights,
        oracle_dtype=str(upcast).removeprefix("torch.") if upcast else None,
    )
    diffs: list[OutputDiff] = []
    for draw in range(draws):
        inputs = workload.inputs(draw)
        expected = cache.get(
            f"{digest.hexdigest()[:24]}-{draw}", partial(_run, reference, inputs, upcast)
        )
        actual = _run(model, inputs)
        torch.cuda.synchronize()
        diffs, error = compare_outputs(expected, actual, rtol, atol)
        if error or any(d.mismatched for d in diffs):
            return result(passed=False, draws=draw + 1, diffs=diffs, error=error)
    return result(passed=True, draws=draws, diffs=diffs)


def _run(model: nn.Module, inputs: list, upcast: torch.dtype | None = None) -> Any:
    if upcast is None:
        return model(*clone(inputs))
    return _cast(model(*_cast(inputs, upcast)), torch.float32)
