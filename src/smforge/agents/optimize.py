"""One optimization stage on one problem: CoVeR against the stage's input kernel.

A candidate is accepted when it passes the static checks and the source audit, keeps the
harness interface, matches the reference on ``ci`` and ``bench-gpu``, launches nothing new that is
not Triton, and beats the input in an interleaved A/B comparison by more than the min-effect band.
If no candidate is accepted, the input carries forward unchanged. Every candidate is saved with
its attempt number and the check that rejected it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Template

from smforge import render
from smforge.agents.analyze import Measurements, measurements_text
from smforge.agents.cover import NO_CODE, cover
from smforge.agents.seed import CHECK_VARIANTS, check_feedback, task_prompt
from smforge.corpus.catalog import Problem
from smforge.harness.audit import audit_interface, audit_launches, audit_source
from smforge.harness.runner import Runner, WorkerError
from smforge.harness.static import validate_source
from smforge.kb.loader import KnowledgeBase
from smforge.llm.client import Completer, Usage
from smforge.models import DeviceProfile, Severity, Stage, Verdict
from smforge.spec import load_spec

PROMPTS = Path(__file__).parent.parent / "llm" / "prompts"
TIMED_VARIANT = "bench-gpu"


@dataclass
class Candidate:
    attempt: int
    outcome: str
    detail: str
    speedup: float | None = None


@dataclass
class StageResult:
    problem: str
    stage: Stage
    accepted: bool
    kernel: Path
    input_us: float
    output_us: float
    attempts: list[Candidate] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def speedup(self) -> float:
        return self.input_us / self.output_us


def system_prompt(
    stage: Stage, profile: DeviceProfile | None, knowledge: str, min_effect: float
) -> str:
    if profile is None:
        device, triton, torch = "an NVIDIA GPU", "3.x", "2.x"
    else:
        d = profile.device
        device = (
            f"{d.name} ({d.arch}), {d.sm_count} SMs, {d.l2_bytes >> 20} MB L2, "
            f"{d.smem_per_block // 1024} KB shared memory per block"
        )
        triton, torch = d.triton, d.torch.split("+")[0]
    return Template((PROMPTS / "optimize.md").read_text()).substitute(
        stage=stage.value,
        instructions=(PROMPTS / "stages" / f"{stage.value}.md").read_text().strip(),
        min_effect=f"{min_effect:.0%}",
        device=device,
        triton=triton,
        torch=torch,
        knowledge=knowledge,
    )


def optimize_stage(
    problem: Problem,
    stage: Stage,
    kernel: Path,
    runner: Runner,
    llm: Completer,
    kb: KnowledgeBase,
    profile: DeviceProfile | None,
    max_iters: int,
    workdir: Path,
    keep: int | None = 2,
    notes: str | None = None,
    measured: Measurements | None = None,
) -> StageResult:
    """Run one stage from ``kernel``; ``measured`` reuses measurements of it already taken."""
    spec = load_spec(problem.spec)
    workdir.mkdir(parents=True, exist_ok=True)
    baseline = kernel.read_text()
    if measured is None:
        args = {"spec": problem.spec, "variant": TIMED_VARIANT}
        measured = Measurements(
            runner.bench(kernel, **args),
            runner.facts(kernel, **args),
            runner.launches(kernel, **args),
            None,
        )
    timing, facts, launches = measured.timing, measured.facts, measured.launches
    sections = [
        task_prompt(problem, spec),
        f"Current kernel (`kernel.py`):\n\n```python\n{baseline.strip()}\n```",
        measurements_text(measured, spec.dtype(spec.variant(TIMED_VARIANT)), profile),
    ]
    if notes:
        sections.append(notes)
    task = "\n\n".join(sections)
    min_effect = runner.config.timing.min_effect
    system = system_prompt(stage, profile, kb.render(stage, [problem.family]), min_effect)
    result = StageResult(problem.id, stage, False, kernel, timing.time_us, timing.time_us)
    candidates: dict[int, Candidate] = {}

    def verify(code: str, attempt: int) -> tuple[bool, str]:
        path = workdir / f"attempt-{attempt}.py"
        path.write_text(code)
        outcome, feedback, comparison = _check(path, code)
        speedup = comparison.speedup if comparison else None
        candidates[attempt] = Candidate(attempt, outcome, feedback.splitlines()[0], speedup)
        if outcome == "accepted":
            result.accepted, result.kernel = True, path
            result.input_us, result.output_us = comparison.baseline_us, comparison.candidate_us
        return outcome == "accepted", feedback

    def _check(path: Path, code: str):
        issues = validate_source(code)
        if not any(issue.check == "syntax_error" for issue in issues):
            issues += audit_source(code, spec.audit.allow) + audit_interface(code, baseline)
        errors = [issue for issue in issues if issue.severity is Severity.ERROR]
        if errors:
            kind = "interface" if all(e.check.startswith("interface") for e in errors) else "static"
            return kind, f"Static checks:\n{render.issues(errors)}", None
        for name in CHECK_VARIANTS:
            if name in spec.variants:
                check = runner.check(path, problem.reference, spec=problem.spec, variant=name)
                if not check.passed:
                    kind = "error" if check.error else "mismatch"
                    return kind, check_feedback(name, check), None
        try:
            new_launches = runner.launches(path, spec=problem.spec, variant=TIMED_VARIANT)
        except WorkerError as e:
            return "error", f"Launch profile failed: {e}", None
        audit = [i for i in audit_launches(new_launches, launches) if i.severity is Severity.ERROR]
        if audit:
            text = f"Launch audit:\n{render.issues(audit)}\n{render.launches(new_launches)}"
            return "launch_audit", text, None
        comparison = runner.compare(kernel, path, spec=problem.spec, variant=TIMED_VARIANT)
        if comparison.verdict is Verdict.FASTER:
            return "accepted", f"Accepted: {render.comparison(comparison)}", comparison
        try:
            deltas = render.fact_deltas(
                facts, runner.facts(path, spec=problem.spec, variant=TIMED_VARIANT)
            )
        except WorkerError as e:
            deltas = f"unavailable: {e}"
        text = (
            f"Correct but not faster: {render.comparison(comparison)}\n\n"
            f"Kernel changes from the current kernel:\n{deltas}\n\n"
            "Try a different configuration and reply with the complete file."
        )
        return str(comparison.verdict), text, comparison

    run = cover(llm, system, task, verify, max_iters, role="optimizer", keep=keep)
    result.usage = run.usage
    result.attempts = [
        candidates.get(i) or Candidate(i, "no_code", NO_CODE) for i in range(len(run.attempts))
    ]
    (workdir / "attempts.json").write_text(
        json.dumps([asdict(a) for a in result.attempts], indent=1)
    )
    (workdir / "output.py").write_text(result.kernel.read_text())
    return result
