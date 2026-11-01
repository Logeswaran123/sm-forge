"""Seed kernels: a straightforward, correct Triton or CUDA C++ kernel for each corpus problem.

A candidate passes when the static checks and the forward-path audit find no error, it matches
the reference on the ``ci`` variant (odd sizes) and on ``bench-gpu``, and its forward launches no
library or ATen compute kernel.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import yaml

from smforge import render
from smforge.agents.cover import CoverResult, Verifier, cover
from smforge.corpus.catalog import Problem
from smforge.harness.audit import audit_launches, audit_source
from smforge.harness.runner import Runner
from smforge.harness.static import validate_source
from smforge.harness.tolerance import resolve_tolerance
from smforge.llm.client import Completer
from smforge.llm.prompting import fill
from smforge.models import CheckResult, DeviceProfile, Dsl, Severity
from smforge.spec import KernelSpec, load_spec

CHECK_VARIANTS = ("ci", "bench-gpu")
ERROR_CHARS = 4000


def system_prompt(profile: DeviceProfile | None, dsl: Dsl = Dsl.TRITON) -> str:
    return fill(dsl, "seed.md", profile)


def task_prompt(problem: Problem, spec: KernelSpec) -> str:
    lines = [
        "Reference (`reference.py`):",
        "",
        f"```python\n{problem.reference.read_text().strip()}\n```",
        "",
        "Variants your kernel is checked on:",
    ]
    for name in CHECK_VARIANTS:
        if name not in spec.variants:
            continue
        variant = spec.variant(name)
        inputs = ", ".join(
            f"{param} {list(shape)} {dtype}"
            for param, shape, dtype in zip(
                variant.params, spec.input_shapes(variant), spec.input_dtypes(variant), strict=True
            )
        )
        rtol, atol = resolve_tolerance(spec.dtype(variant), variant)
        lines.append(
            f"- `{name}`: Model{tuple(spec.init_args(variant))}, inputs {inputs}; "
            f"rtol {rtol:g}, atol {atol:g}"
        )
    lines.append("")
    lines.append("`bench-gpu` is the shape later stages benchmark.")
    return "\n".join(lines)


def verifier(problem: Problem, spec: KernelSpec, runner: Runner, workdir: Path) -> Verifier:
    def verify(code: str, attempt: int) -> tuple[bool, str]:
        path = workdir / f"attempt-{attempt}.py"
        path.write_text(code)
        issues = validate_source(code)
        if not any(issue.check == "syntax_error" for issue in issues):
            issues += audit_source(code, spec.audit.allow)
        errors = [issue for issue in issues if issue.severity is Severity.ERROR]
        if errors:
            return False, f"Static checks:\n{render.issues(errors)}"
        for name in CHECK_VARIANTS:
            if name in spec.variants:
                result = runner.check(path, problem.reference, spec=problem.spec, variant=name)
                if not result.passed:
                    return False, check_feedback(name, result)
        launches = runner.launches(path, spec=problem.spec, variant="bench-gpu")
        audit = [i for i in audit_launches(launches) if i.severity is Severity.ERROR]
        if audit:
            return False, f"Launch audit:\n{render.issues(audit)}\n{render.launches(launches)}"
        return True, "passed"

    return verify


def seed_problem(
    problem: Problem,
    runner: Runner,
    llm: Completer,
    profile: DeviceProfile | None,
    max_iters: int,
    workdir: Path,
) -> CoverResult:
    spec = load_spec(problem.spec)
    workdir.mkdir(parents=True, exist_ok=True)
    result = cover(
        llm,
        system_prompt(profile, problem.dsl),
        task_prompt(problem, spec),
        verifier(problem, spec, runner, workdir),
        max_iters,
        role="seed",
    )
    (workdir / "attempts.json").write_text(
        json.dumps(
            [{"passed": a.passed, "feedback": a.feedback} for a in result.attempts], indent=1
        )
    )
    if result.passed:
        problem.seed.write_text(result.code)
        _write_provenance(problem, llm, result)
    return result


def check_feedback(variant: str, result: CheckResult) -> str:
    text = f"Correctness on the `{variant}` variant:\n{render.check(result)}"
    if result.error:
        text += f"\n\nError:\n{result.error.strip()[-ERROR_CHARS:]}"
    return text


def _write_provenance(problem: Problem, llm: Completer, result: CoverResult) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    usage = result.usage
    record = {
        "model": llm.config.model,
        "effort": llm.config.effort,
        "attempts": len(result.attempts),
        "cost_usd": round(usage.cost_usd, 4),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "commit": commit,
        "date": date.today().isoformat(),
    }
    problem.seed_record.write_text(yaml.safe_dump(record, sort_keys=False))
