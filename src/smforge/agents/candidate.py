"""Checking a candidate kernel against the kernel it would replace.

The checks run cheapest first and stop at the first failure: the static checks and the source
audit, the harness interface (``Model.__init__``, ``get_inputs``, ``get_init_inputs`` unchanged),
correctness on ``ci`` and ``bench-gpu``, float32 correctness on ``ci`` when the base kernel has it,
the launch audit, and an interleaved A/B against the base kernel. A candidate that crashes or
mismatches can be run under compute-sanitizer, and its findings go into the feedback.

A KernelBench problem is float32, and its ``bench-gpu`` and ``ci`` variants run it in float16 or
bfloat16. The float32 check runs ``ci`` with the declared float32 inputs and float32 tolerances,
so a kernel cannot gain speed by rounding float32 operands to TF32 or casting them to float16.
Many seeds already use ``tl.dot`` at its default TF32 precision and fail it, so the check only
applies when the base kernel passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from smforge import render
from smforge.agents.seed import CHECK_VARIANTS, check_feedback
from smforge.corpus.catalog import Problem
from smforge.harness.audit import audit_interface, audit_launches, audit_source
from smforge.harness.runner import Runner, WorkerError
from smforge.harness.sanitizer import SanitizerError, sanitize
from smforge.harness.static import validate_source
from smforge.models import CheckResult, Comparison, FactsReport, LaunchProfile, Severity, Verdict
from smforge.spec import KernelSpec, load_spec

TIMED_VARIANT = "bench-gpu"
FLOAT32_VARIANT = "ci"
_FAULT = re.compile(
    r"CUDA error|illegal memory|misaligned address|unspecified launch failure|worker exited"
)
FLOAT32_ADVICE = (
    "The problem's inputs are float32; the benchmark runs them in a lower precision. The kernel "
    'must stay correct on float32 inputs: give tl.dot input_precision="ieee" (ignored for '
    "float16 operands), and cast to the input's element type, not to tl.float16."
)


@dataclass
class Base:
    """The kernel a candidate competes with, and what is known about it."""

    path: Path
    code: str
    facts: FactsReport
    launches: LaunchProfile
    float32: bool


@dataclass
class Outcome:
    """``kind`` is ``accepted``, a failed check (``static``, ``interface``, ``mismatch``,
    ``error``, ``float32``, ``launch_audit``), or the verdict of a correct candidate that is not
    faster (``parity``, ``slower``)."""

    kind: str
    feedback: str
    comparison: Comparison | None = None

    @property
    def correct(self) -> bool:
        return self.comparison is not None


def float32_applies(spec: KernelSpec) -> bool:
    """Whether ``ci`` runs inputs declared float32 at a lower precision."""
    if FLOAT32_VARIANT not in spec.variants:
        return False
    variant = spec.variant(FLOAT32_VARIANT)
    return spec.declared_dtype(variant) == "float32" and spec.dtype(variant) != "float32"


def check_float32(runner: Runner, kernel: Path, problem: Problem) -> CheckResult:
    return runner.check(
        kernel, problem.reference, spec=problem.spec, variant=FLOAT32_VARIANT, dtype="float32"
    )


def make_base(
    runner: Runner,
    problem: Problem,
    kernel: Path,
    facts: FactsReport | None = None,
    launches: LaunchProfile | None = None,
) -> Base:
    args = {"spec": problem.spec, "variant": TIMED_VARIANT}
    spec = load_spec(problem.spec)
    return Base(
        kernel,
        kernel.read_text(),
        facts or runner.facts(kernel, **args),
        launches or runner.launches(kernel, **args),
        float32_applies(spec) and check_float32(runner, kernel, problem).passed,
    )


def float32_note(base: Base) -> str | None:
    """The line telling the LLM about the float32 check, when it applies."""
    if not base.float32:
        return None
    return (
        f"The current kernel also passes `{FLOAT32_VARIANT}` with float32 inputs, and so must "
        f"yours. {FLOAT32_ADVICE}"
    )


def diagnose(problem: Problem, path: Path) -> str | None:
    """memcheck, then racecheck, on ``ci``: the report of the first that finds errors."""
    variant = "ci" if "ci" in load_spec(problem.spec).variants else None
    for tool in ("memcheck", "racecheck"):
        try:
            report = sanitize(path, tool, problem.spec, variant)
        except SanitizerError:
            return None
        if report.errors:
            return render.sanitizer(report)
    return None


def check_candidate(
    runner: Runner,
    problem: Problem,
    spec: KernelSpec,
    base: Base,
    path: Path,
    sanitize_failures: bool = False,
) -> Outcome:
    """``sanitize_failures`` adds compute-sanitizer findings to a crash or a mismatch."""
    code = path.read_text()
    issues = validate_source(code)
    if not any(issue.check == "syntax_error" for issue in issues):
        issues += audit_source(code, spec.audit.allow) + audit_interface(code, base.code)
    errors = [issue for issue in issues if issue.severity is Severity.ERROR]
    if errors:
        kind = "interface" if all(e.check.startswith("interface") for e in errors) else "static"
        return Outcome(kind, f"Static checks:\n{render.issues(errors)}")
    for name in CHECK_VARIANTS:
        if name in spec.variants:
            check = runner.check(path, problem.reference, spec=problem.spec, variant=name)
            if not check.passed:
                feedback = check_feedback(name, check)
                if sanitize_failures and (not check.error or _FAULT.search(check.error)):
                    if found := diagnose(problem, path):
                        feedback += f"\n\ncompute-sanitizer on `ci`:\n{found}"
                return Outcome("error" if check.error else "mismatch", feedback)
    if base.float32:
        check = check_float32(runner, path, problem)
        if not check.passed:
            text = (
                f"Correctness on `{FLOAT32_VARIANT}` with float32 inputs, which the current "
                f"kernel passes:\n{render.check(check)}\n\n{FLOAT32_ADVICE}"
            )
            if check.error:
                text += f"\n\nError:\n{check.error.strip()[-2000:]}"
            return Outcome("float32", text)
    try:
        new_launches = runner.launches(path, spec=problem.spec, variant=TIMED_VARIANT)
    except WorkerError as e:
        return Outcome("error", f"Launch profile failed: {e}")
    audit = [i for i in audit_launches(new_launches, base.launches) if i.severity is Severity.ERROR]
    if audit:
        text = f"Launch audit:\n{render.issues(audit)}\n{render.launches(new_launches)}"
        return Outcome("launch_audit", text)
    comparison = runner.compare(base.path, path, spec=problem.spec, variant=TIMED_VARIANT)
    if comparison.verdict is Verdict.FASTER:
        return Outcome("accepted", f"Accepted: {render.comparison(comparison)}", comparison)
    try:
        deltas = render.fact_deltas(
            base.facts, runner.facts(path, spec=problem.spec, variant=TIMED_VARIANT)
        )
    except WorkerError as e:
        deltas = f"unavailable: {e}"
    text = (
        f"Correct but not faster: {render.comparison(comparison)}\n\n"
        f"Kernel changes from the current kernel:\n{deltas}"
    )
    return Outcome(str(comparison.verdict), text, comparison)
