"""One optimization stage on one problem: CoVeR against the stage's input kernel.

A candidate is accepted when it passes the checks in ``smforge.agents.candidate`` and beats the
input in an interleaved A/B comparison by more than the min-effect band. If no candidate is
accepted, the input carries forward unchanged. Every candidate is saved with
its attempt number and the check that rejected it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Template

from smforge.agents.analyze import Measurements, measurements_text
from smforge.agents.candidate import TIMED_VARIANT, check_candidate, float32_note, make_base
from smforge.agents.cover import NO_CODE, cover
from smforge.agents.seed import task_prompt
from smforge.corpus.catalog import Problem
from smforge.harness.runner import Runner
from smforge.kb.loader import KnowledgeBase
from smforge.llm.client import Completer, Usage
from smforge.models import DeviceProfile, Stage
from smforge.spec import load_spec

PROMPTS = Path(__file__).parent.parent / "llm" / "prompts"


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
    if measured is None:
        args = {"spec": problem.spec, "variant": TIMED_VARIANT}
        measured = Measurements(
            runner.bench(kernel, **args),
            runner.facts(kernel, **args),
            runner.launches(kernel, **args),
            None,
        )
    timing = measured.timing
    base = make_base(runner, problem, kernel, measured.facts, measured.launches)
    sections = [
        task_prompt(problem, spec),
        f"Current kernel (`kernel.py`):\n\n```python\n{base.code.strip()}\n```",
        measurements_text(measured, spec.dtype(spec.variant(TIMED_VARIANT)), profile),
    ]
    if note := float32_note(base):
        sections.append(note)
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
        outcome = check_candidate(runner, problem, spec, base, path)
        comparison = outcome.comparison
        speedup = comparison.speedup if comparison else None
        candidates[attempt] = Candidate(
            attempt, outcome.kind, outcome.feedback.splitlines()[0], speedup
        )
        feedback = outcome.feedback
        if outcome.kind == "accepted":
            result.accepted, result.kernel = True, path
            result.input_us, result.output_us = comparison.baseline_us, comparison.candidate_us
        elif outcome.correct:
            feedback += "\n\nTry a different configuration and reply with the complete file."
        return outcome.kind == "accepted", feedback

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
