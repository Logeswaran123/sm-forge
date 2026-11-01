"""Running each entry's kernel pair on the GPU (parent side).

Every kernel that should pass goes through what a candidate faces: the static validator, the
source audit, correctness on ``ci`` and the timed variant, and the launch audit. A pattern passes
when its example's after kernel also beats the before kernel in an interleaved A/B comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from smforge import render
from smforge.corpus.catalog import Problem
from smforge.harness.audit import audit_launches, audit_source
from smforge.harness.runner import Runner, WorkerError
from smforge.harness.static import validate_source
from smforge.kb.loader import KnowledgeBase
from smforge.kb.schema import Constraint, Example, Measurement
from smforge.models import DeviceInfo, Severity, Verdict
from smforge.spec import load_spec


@dataclass
class Outcome:
    id: str
    kind: str
    passed: bool
    detail: str
    measurement: Measurement | None = None


def verify_constraint(
    kb: KnowledgeBase,
    constraint: Constraint,
    runner: Runner,
    problems: dict[str, Problem],
    dev: set[str],
) -> Outcome:
    def outcome(passed: bool, detail: str) -> Outcome:
        return Outcome(constraint.id, "constraint", passed, detail)

    check = constraint.check
    problem = problems.get(check.problem)
    if problem is None:
        return outcome(False, f"unknown problem {check.problem}")
    if check.problem not in dev:
        return outcome(False, f"{check.problem} is not in the dev split")
    wrong, correct = kb.check_files(constraint)
    variants = [check.variant] if check.expect != "slower" else ["ci", check.variant]
    reason = kernel_fails(correct, problem, runner, variants)
    if reason:
        return outcome(False, f"correct.py: {reason}")
    if check.expect == "static":
        code = wrong.read_text()
        found = {issue.check for issue in validate_source(code) + audit_source(code)}
        if check.match not in found:
            return outcome(False, f"wrong.py: no {check.match} issue, got {sorted(found)}")
        return outcome(True, f"wrong.py reports {check.match}")
    if check.expect == "slower":
        comparison = runner.compare(wrong, correct, spec=problem.spec, variant=check.variant)
        if comparison.verdict is not Verdict.FASTER:
            return outcome(False, f"correct.py is {comparison.verdict} ({comparison.speedup:.2f}x)")
        return outcome(True, f"correct.py is {comparison.speedup:.2f}x faster")
    result = runner.check(wrong, problem.reference, spec=problem.spec, variant=check.variant)
    if result.passed:
        return outcome(False, "wrong.py passes the correctness check")
    if check.expect == "error":
        if not result.error or (check.match and check.match not in result.error):
            return outcome(False, f"wrong.py: expected an error with {check.match!r}")
        return outcome(True, f"wrong.py fails: {result.error.strip().splitlines()[-1][:120]}")
    if result.error:
        return outcome(False, f"wrong.py: expected a mismatch, got {result.error.splitlines()[0]}")
    worst = max(result.diffs, key=lambda d: d.max_abs)
    return outcome(True, f"wrong.py mismatches, max abs {worst.max_abs:.3g}")


def verify_example(
    kb: KnowledgeBase,
    example: Example,
    runner: Runner,
    problems: dict[str, Problem],
    dev: set[str],
    device: DeviceInfo,
) -> Outcome:
    def outcome(passed: bool, detail: str, measurement: Measurement | None = None) -> Outcome:
        return Outcome(example.id, "example", passed, detail, measurement)

    problem = problems.get(example.problem)
    if problem is None:
        return outcome(False, f"unknown problem {example.problem}")
    if example.problem not in dev:
        return outcome(False, f"{example.problem} is not in the dev split")
    before, after = kb.example_files(example)
    variants = ["ci", example.variant]
    for path in (before, after):
        reason = kernel_fails(path, problem, runner, variants)
        if reason:
            return outcome(False, f"{path.name}: {reason}")
    comparison = runner.compare(before, after, spec=problem.spec, variant=example.variant)
    measurement = Measurement(
        gpu=device.slug,
        variant=example.variant,
        before_us=round(comparison.baseline_us, 2),
        after_us=round(comparison.candidate_us, 2),
        speedup=round(comparison.speedup, 3),
        triton=device.triton,
        cuda=device.cuda,
        date=date.today().isoformat(),
    )
    detail = (
        f"{comparison.baseline_us:.1f} -> {comparison.candidate_us:.1f} us "
        f"({comparison.speedup:.2f}x, {comparison.verdict})"
    )
    return outcome(comparison.verdict is Verdict.FASTER, detail, measurement)


def kernel_fails(path: Path, problem: Problem, runner: Runner, variants: list[str]) -> str | None:
    """Why a kernel that should pass does not, or None."""
    code = path.read_text()
    spec = load_spec(problem.spec)
    issues = validate_source(code)
    if not any(issue.check == "syntax_error" for issue in issues):
        issues += audit_source(code, spec.audit.allow)
    errors = [issue for issue in issues if issue.severity is Severity.ERROR]
    if errors:
        return f"static checks\n{render.issues(errors)}"
    for variant in dict.fromkeys(variants):
        if variant not in spec.variants:
            continue
        result = runner.check(path, problem.reference, spec=problem.spec, variant=variant)
        if not result.passed:
            return f"{variant}: {render.check(result)}"
    try:
        launches = runner.launches(path, spec=problem.spec, variant=variants[-1])
    except WorkerError as e:
        return f"launch audit: {e}"
    audit = [i for i in audit_launches(launches) if i.severity is Severity.ERROR]
    if audit:
        return f"launch audit\n{render.issues(audit)}"
    return None
