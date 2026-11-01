"""Claude Code workspace engine: one problem, one headless Claude Code session, one trial tree.

``build_workspace`` writes a directory holding the kernel (``t0``), read-only copies of the
reference and spec, the knowledge base filtered for the device, the stage instructions, the
problem's lessons ledger, a ``CLAUDE.md`` with the rules and the measurements of ``t0``, an
``/optimize-kernel`` command, and project settings that let Claude Code write only under
``work/`` and run only ``smforge-skill``. The session saves its candidates with
``smforge-skill trial save``, which checks and times each one against ``t0``.

``run_claude`` runs the session with a turn cap, then finalizes the trial tree itself (the
session's claims are not used): the best trial is checked again with the pipeline's candidate
checks in this process's worker, against the corpus copy of the reference and spec, then
confirmed (memcheck, and the A/B against the seed in fresh workers, the slowest counting).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

import yaml

from smforge.agents.analyze import (
    TIMED_VARIANT,
    issues_text,
    measure,
    measurements_text,
    pre_analyze,
)
from smforge.agents.candidate import (
    Confirmation,
    check_candidate,
    confirm,
    float32_note,
    make_base,
)
from smforge.agents.optimize import PROMPTS
from smforge.agents.seed import task_prompt
from smforge.config import Config
from smforge.corpus.catalog import Problem
from smforge.harness.runner import Runner
from smforge.kb.loader import KnowledgeBase
from smforge.lessons import read as read_lessons
from smforge.models import DeviceProfile, Stage
from smforge.spec import load_spec
from smforge.trials import TrialTree

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"
SESSION_TIMEOUT = 4 * 3600
TOOLS = "Bash,Read,Edit,Write,Glob,Grep"
SKILL_COMMANDS = (
    "validate",
    "check",
    "bench",
    "facts",
    "sanitize",
    "profile",
    "trial save",
    "trial status",
    "trial best",
)
READ_ONLY = ("CLAUDE.md", "config.yaml", "kernel.py", "kb.md", "stages.md", "lessons.md")
PERMISSIONS = Path(".claude") / "permissions.json"


class ClaudeSessionError(RuntimeError):
    pass


@dataclass
class Session:
    """What the headless session reported about itself."""

    stop: str
    turns: int
    cost_usd: float
    summary: str
    error: bool


@dataclass
class ClaudeResult:
    problem: str
    workspace: Path
    session: Session
    trials: int
    final: str | None
    seed_us: float
    final_us: float
    trial_speedup: float | None = None
    ab: list[float] = field(default_factory=list)
    kernel: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        return self.seed_us / self.final_us


def build_workspace(
    root: Path,
    problem: Problem,
    runner: Runner,
    kb: KnowledgeBase,
    profile: DeviceProfile | None,
    ledger: Path | None,
    max_trials: int,
    ncu: bool = True,
) -> TrialTree:
    """Write the workspace and measure ``t0``; returns the trial tree."""
    root = root.resolve()
    if root.exists():
        raise ClaudeSessionError(f"{root} exists; a workspace is built once")
    (root / "problem").mkdir(parents=True)
    (root / "work").mkdir()
    for name in ("reference.py", "spec.yaml"):
        shutil.copyfile(problem.path / name, root / "problem" / name)
    shutil.copyfile(problem.seed, root / "kernel.py")
    spec = load_spec(problem.spec)
    dtype = spec.dtype(spec.variant(TIMED_VARIANT))
    measured = measure(runner, problem.seed, problem, ncu)
    issues = pre_analyze(problem.seed.read_text(), measured, dtype, profile)
    tree = TrialTree.create(root / "trials", problem, problem.seed, runner, ledger)
    float32 = float32_note(tree.base())
    device, triton, torch = "an NVIDIA GPU", "3.x", "2.x"
    if profile is not None:
        d = profile.device
        device = (
            f"{d.name} ({d.arch}), {d.sm_count} SMs, {d.l2_bytes >> 20} MB L2, "
            f"{d.smem_per_block // 1024} KB shared memory per block"
        )
        triton, torch = d.triton, d.torch.split("+")[0]
    task = (
        task_prompt(problem, spec)
        .replace("(`reference.py`)", "(`problem/reference.py`)")
        .replace("is the shape later stages benchmark", "is the shape trials are timed on")
    )
    if float32:
        task += f"\n\n{float32}"
    min_effect = runner.config.timing.min_effect
    (root / "CLAUDE.md").write_text(
        Template((TEMPLATES / "CLAUDE.md").read_text()).substitute(
            problem=problem.id,
            device=device,
            triton=triton,
            torch=torch,
            task=task,
            measurements=measurements_text(measured, dtype, profile),
            issues=issues_text(issues),
            float32_check=", and on `ci` with float32 inputs" if float32 else "",
            min_effect=f"{min_effect:.0%}",
        )
    )
    (root / "kb.md").write_text("# Knowledge base\n\n" + kb.render_all([problem.family]) + "\n")
    (root / "stages.md").write_text(
        "# Optimization stages\n\n"
        + "\n".join(
            f"## {stage}\n\n{(PROMPTS / 'stages' / f'{stage}.md').read_text().strip()}\n"
            for stage in Stage
        )
    )
    past = read_lessons(ledger, lines=200)
    (root / "lessons.md").write_text(
        f"# Lessons: {problem.id}\n\n" + (past or "No earlier runs on this GPU.") + "\n"
    )
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "problem": problem.id,
                "family": problem.family,
                "device": device,
                "timed_variant": TIMED_VARIANT,
                "min_effect": min_effect,
                "max_trials": max_trials,
                "t0_us": round(tree.state.baseline_us, 2),
                "trials": "trials",
            },
            sort_keys=False,
        )
    )
    commands = root / ".claude" / "commands"
    commands.mkdir(parents=True)
    (commands / "optimize-kernel.md").write_text(
        Template((TEMPLATES / "optimize-kernel.md").read_text()).substitute(max_trials=max_trials)
    )
    (root / PERMISSIONS).write_text(json.dumps(settings(root), indent=2))
    for name in READ_ONLY:
        _read_only(root / name)
    for path in (root / "problem").iterdir():
        _read_only(path)
    return tree


def settings(root: Path) -> dict:
    """Permissions for the session: write only under ``work/``, run only ``smforge-skill``.

    They are passed with ``--settings``: Claude Code ignores allow rules in the project settings
    of a directory nobody has trusted interactively. Auto memory is off, so the session neither
    reads nor writes memories kept for other work.
    """
    allow = ["Read", "Glob", "Grep", f"Edit(/{root}/work/**)"]
    allow += [f"Bash(smforge-skill {command} *)" for command in SKILL_COMMANDS]
    deny = [f"Edit(/{root}/{name})" for name in READ_ONLY]
    deny += [f"Edit(/{root}/problem/**)", f"Edit(/{root}/trials/**)", f"Edit(/{root}/.claude/**)"]
    deny += ["Bash(smforge-skill trial init *)", "Bash(smforge-skill trial finalize *)"]
    return {
        "permissions": {"allow": allow, "deny": deny, "defaultMode": "dontAsk"},
        "autoMemoryEnabled": False,
    }


def run_session(root: Path, config: Config, max_turns: int, budget_usd: float | None) -> Session:
    """Run ``/optimize-kernel`` headless in ``root``; the transcript goes to ``transcript.jsonl``."""
    binary = config.llm.claude_bin or shutil.which("claude")
    if binary is None:
        raise ClaudeSessionError("claude not found; install Claude Code or set CLAUDE_BIN")
    command = [
        binary,
        "-p",
        "/optimize-kernel",
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-turns",
        str(max_turns),
        "--model",
        config.llm.model,
        "--effort",
        config.llm.effort_for("optimizer"),
        "--tools",
        TOOLS,
        "--settings",
        str(root / PERMISSIONS),
        "--permission-mode",
        "dontAsk",
        "--setting-sources",
        "project",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]
    if budget_usd is not None:
        command += ["--max-budget-usd", str(budget_usd)]
    env = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
        "CACHE_DIR": str(config.cache_dir.resolve()),
        "HARDWARE_DIR": str(config.hardware_dir.resolve()),
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    }
    with (root / "transcript.jsonl").open("w") as transcript:
        try:
            subprocess.run(
                command,
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=transcript,
                stderr=subprocess.STDOUT,
                timeout=SESSION_TIMEOUT,
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeSessionError(f"session ran past {SESSION_TIMEOUT}s") from e
    return parse_session(root / "transcript.jsonl")


def parse_session(path: Path) -> Session:
    result = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
    if result is None:
        tail = path.read_text().strip()[-2000:]
        raise ClaudeSessionError(f"the session ended without a result: {tail}")
    return Session(
        stop=result.get("subtype", ""),
        turns=result.get("num_turns", 0),
        cost_usd=result.get("total_cost_usd", 0.0),
        summary=str(result.get("result", "")).strip(),
        error=bool(result.get("is_error")),
    )


def run_claude(
    problem: Problem,
    runner: Runner,
    kb: KnowledgeBase,
    profile: DeviceProfile | None,
    config: Config,
    workdir: Path,
    max_trials: int,
    max_turns: int,
    budget_usd: float | None = None,
    ledger: Path | None = None,
    ncu: bool = True,
    memcheck: bool = True,
) -> ClaudeResult:
    tree = build_workspace(workdir, problem, runner, kb, profile, ledger, max_trials, ncu)
    root = tree.root.parent
    session = run_session(root, config, max_turns, budget_usd)
    tree = TrialTree(root / "trials")
    if session.error and len(tree.trials) == 1:
        raise ClaudeSessionError(f"the session failed before its first trial: {session.summary}")
    spec = load_spec(problem.spec)
    base = make_base(runner, problem, problem.seed)
    checks: dict[str, Confirmation] = {}

    def verify(trial, path: Path) -> str | None:
        """The candidate checks again in this worker, then memcheck and repeated A/B runs."""
        outcome = check_candidate(runner, problem, spec, base, path)
        if not outcome.correct:
            return f"re-check {outcome.kind}: {outcome.feedback.splitlines()[0]}"
        checks[trial.id] = confirm(runner, problem, problem.seed, path, memcheck)
        return None if checks[trial.id].ok else checks[trial.id].reason()

    output = root / "output.py"
    final = tree.finalize(output, verify)
    t0_us = tree.state.baseline_us
    result = ClaudeResult(problem.id, root, session, len(tree.trials) - 1, None, t0_us, t0_us)
    if session.error:
        result.notes.append(f"the session ended with an error: {session.summary}")
    for trial in tree.trials.values():
        if trial.final_check not in (None, "ok"):
            result.notes.append(f"{trial.id} not finalized: {trial.final_check}")
    if final is None:
        shutil.copyfile(problem.seed, output)
        return result
    check = checks[final.id]
    result.final, result.kernel = final.id, output
    result.trial_speedup, result.ab = final.speedup, [c.speedup for c in check.comparisons]
    result.seed_us, result.final_us = check.worst.baseline_us, check.worst.candidate_us
    return result


def _read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
