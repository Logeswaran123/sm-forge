"""Trial tree for one problem: every candidate kernel, the trial it came from, and its measurements.

``t0`` is the baseline. Saving a trial checks it at once against ``t0`` with the checks a
pipeline candidate goes through (``smforge.agents.candidate``), A/B timing included, so every
result in the tree was measured by the harness and none can be entered by hand. Each trial's file
is stored with the sha256 of its source; one whose file has changed since does not rank.

Ranking: correct trials only, by speedup over ``t0``, parity at 1.0; on equal rank the faster
measured time wins. A trial that failed a check, or whose check never finished, does not rank.
``finalize`` takes the best trial that is faster than ``t0``, runs memcheck on it when asked, and
moves to the next one if memcheck finds errors. It refuses parity and regressions: when no trial
is faster, nothing is finalized.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from smforge import lessons
from smforge.agents.candidate import TIMED_VARIANT, Base, Outcome, check_candidate, make_base
from smforge.corpus.catalog import Problem
from smforge.harness.runner import Runner
from smforge.models import FactsReport, LaunchProfile, SanitizerReport
from smforge.spec import load_spec

STATE = "trials.json"


class Trial(BaseModel):
    id: str
    parent: str | None
    file: str
    sha256: str
    strategy: str = ""
    outcome: str | None = None
    detail: str = ""
    speedup: float | None = None
    time_us: float | None = None
    baseline_us: float | None = None
    memcheck: str | None = None
    created: str = ""

    @property
    def correct(self) -> bool:
        return self.outcome in ("baseline", "accepted", "parity", "slower")


class TrialState(BaseModel):
    problem_id: str
    source: str
    family: str
    problem_path: Path
    baseline_us: float
    facts: FactsReport
    launches: LaunchProfile
    float32: bool
    ledger: Path | None = None
    trials: dict[str, Trial] = Field(default_factory=dict)
    final: str | None = None


class TrialError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrialTree:
    def __init__(self, root: Path):
        self.root = root
        path = root / STATE
        if not path.exists():
            raise TrialError(f"no trial tree in {root}; run `smforge-skill trial init` first")
        self.state = TrialState.model_validate_json(path.read_text())
        state = self.state
        self.problem = Problem(
            id=state.problem_id, source=state.source, family=state.family, path=state.problem_path
        )

    @classmethod
    def create(
        cls,
        root: Path,
        problem: Problem,
        baseline: Path,
        runner: Runner,
        ledger: Path | None = None,
    ) -> TrialTree:
        """Measure ``baseline`` and store it as ``t0``; ``ledger`` gets a line per trial."""
        if (root / STATE).exists():
            raise TrialError(f"{root} already holds a trial tree")
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline, root / "t0.py")
        base = make_base(runner, problem, root / "t0.py")
        timing = runner.bench(root / "t0.py", spec=problem.spec, variant=TIMED_VARIANT)
        state = TrialState(
            problem_id=problem.id,
            source=problem.source,
            family=problem.family,
            problem_path=problem.path.resolve(),
            baseline_us=timing.time_us,
            facts=base.facts,
            launches=base.launches,
            float32=base.float32,
            ledger=ledger.resolve() if ledger else None,
        )
        state.trials["t0"] = Trial(
            id="t0",
            parent=None,
            file="t0.py",
            sha256=sha256(root / "t0.py"),
            outcome="baseline",
            speedup=1.0,
            time_us=timing.time_us,
            baseline_us=timing.time_us,
            created=_now(),
        )
        (root / STATE).write_text(state.model_dump_json(indent=1))
        if ledger is not None:
            lessons.start(ledger, f"trials, t0 {timing.time_us:.1f} us")
        return cls(root)

    @property
    def trials(self) -> dict[str, Trial]:
        return self.state.trials

    def path(self, trial: Trial) -> Path:
        return self.root / trial.file

    def base(self) -> Base:
        t0 = self.path(self.trials["t0"])
        return Base(t0, t0.read_text(), self.state.facts, self.state.launches, self.state.float32)

    def write(self) -> None:
        (self.root / STATE).write_text(self.state.model_dump_json(indent=1))

    def save(self, source: Path, parent: str | None = "t0", strategy: str = "") -> Trial:
        if parent is not None and parent not in self.trials:
            raise TrialError(f"no trial {parent}; the tree has {', '.join(self.trials)}")
        trial_id = f"t{len(self.trials)}"
        shutil.copyfile(source, self.root / f"{trial_id}.py")
        trial = Trial(
            id=trial_id,
            parent=parent,
            file=f"{trial_id}.py",
            sha256=sha256(self.root / f"{trial_id}.py"),
            strategy=strategy,
            created=_now(),
        )
        self.trials[trial_id] = trial
        self.write()
        return trial

    def record(self, trial: Trial, outcome: Outcome) -> Trial:
        trial.outcome = outcome.kind
        trial.detail = outcome.feedback.splitlines()[0]
        if outcome.comparison is not None:
            trial.speedup = outcome.comparison.speedup
            trial.baseline_us = outcome.comparison.baseline_us
            trial.time_us = outcome.comparison.candidate_us
        self.write()
        return trial

    def evaluate(self, trial: Trial, runner: Runner, sanitize_failures: bool = False) -> Outcome:
        """Check ``trial`` against ``t0`` and record what the checks found."""
        spec = load_spec(self.problem.spec)
        outcome = check_candidate(
            runner, self.problem, spec, self.base(), self.path(trial), sanitize_failures
        )
        self.record(trial, outcome)
        if self.state.ledger is not None:
            lessons.add(
                self.state.ledger,
                f"{trial.id} (from {trial.parent}): {trial.outcome}"
                f"{lessons.speedup(trial.speedup, 't0')}: {trial.strategy or trial.detail}",
            )
        return outcome

    def rank(self, trial: Trial) -> tuple[float, float] | None:
        if trial.id == "t0" or not trial.correct or trial.time_us is None:
            return None
        if sha256(self.path(trial)) != trial.sha256:
            return None
        return (trial.speedup if trial.outcome != "parity" else 1.0, -trial.time_us)

    def ranked(self) -> list[Trial]:
        ranked = [(rank, t) for t in self.trials.values() if (rank := self.rank(t)) is not None]
        return [t for _, t in sorted(ranked, key=lambda item: item[0], reverse=True)]

    def best(self) -> Trial | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def finalize(
        self, output: Path, memcheck: Callable[[Path], SanitizerReport] | None = None
    ) -> Trial | None:
        """Copy the best trial faster than ``t0`` that memcheck passes to ``output``."""
        self.state.final = None
        for trial in self.ranked():
            if trial.outcome != "accepted":
                break
            if memcheck is not None and trial.memcheck is None:
                report = memcheck(self.path(trial))
                trial.memcheck = "ok" if report.ok else f"{report.errors} errors"
                self.write()
            if trial.memcheck not in (None, "ok"):
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path(trial), output)
            self.state.final = trial.id
            self.write()
            if self.state.ledger is not None:
                lessons.add(
                    self.state.ledger, f"finalized {trial.id}, {trial.speedup:.2f}x over t0"
                )
            return trial
        self.write()
        return None

    def status(self) -> str:
        best = self.best()
        lines = [
            f"Trial tree: {self.problem.id}, t0 {self.state.baseline_us:.1f} us on {TIMED_VARIANT}",
            f"best: {best.id} {best.speedup:.2f}x" if best else "best: none faster than t0 yet",
        ]
        children: dict[str | None, list[str]] = {}
        for trial in self.trials.values():
            children.setdefault(trial.parent, []).append(trial.id)

        def walk(trial_id: str, prefix: str, last: bool) -> None:
            trial = self.trials[trial_id]
            result = trial.outcome or "unchecked"
            if trial.speedup is not None and trial.id != "t0":
                result += f" {trial.speedup:.2f}x"
            if trial.memcheck:
                result += f", memcheck {trial.memcheck}"
            mark = " <- best" if best and trial_id == best.id else ""
            strategy = f" | {trial.strategy[:70]}" if trial.strategy else ""
            lines.append(
                f"{prefix}{'└── ' if last else '├── '}{trial_id}: {result}{strategy}{mark}"
            )
            kids = children.get(trial_id, [])
            for i, kid in enumerate(kids):
                walk(kid, prefix + ("    " if last else "│   "), i == len(kids) - 1)

        roots = children.get(None, [])
        for i, root in enumerate(roots):
            walk(root, "", i == len(roots) - 1)
        return "\n".join(lines)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
