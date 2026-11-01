"""Finding what is wrong with a kernel: fact-backed issues, then the analyzer's judgment.

``pre_analyze`` reads the measurements (SASS facts, launches, ncu counters) and the source and
emits issues that the numbers prove. ``analyze`` gives those, the kernel and the reference to the
LLM, which adds issues the numbers cannot show (an algebraic shortcut, unfused ops). The LLM
cannot drop a fact-backed issue.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from smforge import render
from smforge.agents.seed import task_prompt
from smforge.corpus.catalog import Problem
from smforge.device.tools import tool_status
from smforge.harness.ncu import NcuError, NcuUnavailable, profile_ncu
from smforge.harness.runner import Runner
from smforge.harness.source import KernelSource
from smforge.harness.static import validate_source
from smforge.llm.client import Completer, Usage
from smforge.llm.structured import ask
from smforge.models import (
    DeviceProfile,
    FactsReport,
    Issue,
    IssueType,
    LaunchProfile,
    NcuKernel,
    Timing,
)
from smforge.spec import load_spec

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent.parent / "llm" / "prompts"
TIMED_VARIANT = "bench-gpu"
MIN_BYTES_PER_SECTOR = 24.0
MAX_CONFLICTS_PER_WAVEFRONT = 0.25
LAUNCH_BOUND_US = 10.0
IDLE_SHARE = 0.5


@dataclass
class Measurements:
    timing: Timing
    facts: FactsReport
    launches: LaunchProfile
    ncu: list[NcuKernel] | None


class AnalyzerIssue(BaseModel):
    type: IssueType
    evidence: str
    kernel: str | None


class AnalyzerReply(BaseModel):
    summary: str
    issues: list[AnalyzerIssue]


def measure(runner: Runner, kernel: Path, problem: Problem, ncu: bool = True) -> Measurements:
    """Timing, facts and launches, and ncu counters when ``ncu``.

    When ncu fails, the tool probe runs once more: if it fails too, ncu is broken for every
    program (counter access was withdrawn mid-run, say) and NcuUnavailable stops the caller;
    otherwise only this kernel goes without counters.
    """
    args = {"spec": problem.spec, "variant": TIMED_VARIANT}
    facts = runner.facts(kernel, **args)
    counters = None
    if ncu:
        try:
            names = {k.name for k in facts.kernels}
            counters = profile_ncu(kernel, names, **args) if names else []
        except NcuError as e:
            status = tool_status("ncu")
            if not status.ok:
                raise NcuUnavailable(f"ncu stopped working: {status.detail}") from e
            log.warning("ncu failed on %s: %s", kernel, e)
    return Measurements(
        runner.bench(kernel, **args), facts, runner.launches(kernel, **args), counters
    )


def pre_analyze(
    code: str, m: Measurements, dtype: str, profile: DeviceProfile | None
) -> list[Issue]:
    """Issues the facts, the ncu counters or the source prove."""
    issues: list[Issue] = []

    def add(kind: IssueType, evidence: str, kernel: str | None = None, source: str = "facts"):
        issues.append(Issue(type=kind, evidence=evidence, kernel=kernel, source=source))

    src = KernelSource(code)
    sms = profile.device.sm_count if profile else None
    for k in m.facts.kernels:
        fn = src.triton_functions.get(k.name)
        dots = [c for c, path in src.calls(fn) if path == "triton.language.dot"] if fn else []
        if k.n_spills:
            add(
                IssueType.REGISTER_SPILL, f"{k.n_spills} spilled registers, {k.n_regs} used", k.name
            )
        if dots and not k.mma:
            add(IssueType.NO_TENSOR_CORE_MMA, "tl.dot compiled without MMA instructions", k.name)
        if dots and k.num_stages <= 1 and "LDGSTS" not in k.instructions and k.mma:
            add(IssueType.MISSING_PIPELINING, "num_stages=1, no cp.async (LDGSTS) in SASS", k.name)
        if k.ctas_per_sm is not None and k.ctas_per_sm * k.num_warps < 8:
            add(
                IssueType.LOW_OCCUPANCY,
                f"{k.ctas_per_sm} CTA/SM x {k.num_warps} warps resident",
                k.name,
            )
        if sms and k.ctas < sms:
            add(IssueType.GRID_UNDERFILL, f"{k.ctas} CTAs for {sms} SMs", k.name)
        elif k.waves and 1 < k.waves < 4 and 0 < k.waves % 1 < 0.5:
            add(IssueType.TAIL_WAVE, f"{k.waves:.2f} waves: the last is mostly idle", k.name)
        if dtype == "float32" and any(
            not any(kw.arg == "input_precision" for kw in call.keywords) for call in dots
        ):
            add(
                IssueType.TF32_PRECISION_RISK,
                "fp32 tl.dot without input_precision defaults to TF32",
                k.name,
                "source",
            )
    if src.triton_functions and not _autotuned(src):
        add(IssueType.UNTUNED_CONFIG, "no @triton.autotune; one hard-coded config", None, "source")
    for issue in validate_source(code):
        if issue.check == "host_sync":
            add(IssueType.HOST_SYNC_IN_FORWARD, issue.message, None, "source")
        elif issue.check == "smem_estimate":
            add(IssueType.SMEM_OVER_BUDGET, issue.message, None, "source")
    calls = sum(launch.calls for launch in m.launches.launches)
    idle, span = m.launches.idle_us, m.launches.span_us
    if calls >= 2 and idle and span and idle > IDLE_SHARE * span:
        add(
            IssueType.LAUNCH_BOUND,
            f"{calls:g} launches per forward; the GPU is idle {idle:.1f} of the {span:.1f} us "
            "from the first kernel's start to the last one's end",
        )
    elif calls >= 4 and m.launches.kernel_us / calls < LAUNCH_BOUND_US:
        add(
            IssueType.LAUNCH_BOUND,
            f"{calls:g} launches per forward averaging {m.launches.kernel_us / calls:.1f} us",
        )
    for k in m.ncu or []:
        if k.load_bytes_per_sector and k.load_bytes_per_sector < MIN_BYTES_PER_SECTOR:
            add(
                IssueType.UNCOALESCED_ACCESS,
                f"global loads use {k.load_bytes_per_sector:.1f} of 32 bytes per sector",
                k.name,
                "ncu",
            )
        conflicts = k.shared_conflicts_per_wavefront
        if conflicts and conflicts > MAX_CONFLICTS_PER_WAVEFRONT:
            add(
                IssueType.BANK_CONFLICT,
                f"{conflicts:.2f} shared-memory bank conflicts per load wavefront",
                k.name,
                "ncu",
            )
    return issues


def _autotuned(src: KernelSource) -> bool:
    for fn in src.triton_functions.values():
        for decorator in fn.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if (src.dotted(target) or "").endswith("autotune"):
                return True
    return False


def analyze(
    llm: Completer,
    problem: Problem,
    code: str,
    m: Measurements,
    found: list[Issue],
    profile: DeviceProfile | None,
) -> tuple[str, list[Issue], Usage]:
    """The analyzer's summary and the full issue list: ``found`` plus what the LLM adds."""
    spec = load_spec(problem.spec)
    system = (PROMPTS / "analyze.md").read_text()
    task = "\n\n".join(
        [
            task_prompt(problem, spec),
            f"Current kernel (`kernel.py`):\n\n```python\n{code.strip()}\n```",
            measurements_text(m, spec.dtype(spec.variant(TIMED_VARIANT)), profile),
            "Issues already established from these measurements:\n" + issues_text(found),
        ]
    )
    reply, raw = ask(llm, system, [{"role": "user", "content": task}], AnalyzerReply, "analyzer")
    usage = Usage()
    usage.add(raw.usage)
    issues = list(found)
    seen = {(i.type, i.kernel) for i in found}
    for issue in reply.issues:
        if (issue.type, issue.kernel) not in seen:
            seen.add((issue.type, issue.kernel))
            issues.append(Issue(**issue.model_dump(), source="llm"))
    return reply.summary, issues, usage


def measurements_text(m: Measurements, dtype: str, profile: DeviceProfile | None) -> str:
    ridge = profile.roofline.ridge(dtype) if profile and profile.roofline else None
    lines = [
        f"Measured on `{TIMED_VARIANT}`: {render.timing(m.timing)}",
        "",
        render.facts(m.facts, ridge),
        "",
        render.launches(m.launches),
    ]
    if m.ncu:
        lines += ["", "Nsight Compute:"]
        for k in m.ncu:
            line = f"  {k.name}: SM {k.sm_pct:.0f}%, DRAM {k.dram_pct:.0f}% of peak"
            if k.load_bytes_per_sector:
                line += f", global loads {k.load_bytes_per_sector:.1f}/32 B per sector"
            if k.shared_conflicts_per_wavefront is not None:
                line += f", {k.shared_conflicts_per_wavefront:.2f} bank conflicts per wavefront"
            lines.append(line)
    return "\n".join(lines)


def issues_text(issues: list[Issue]) -> str:
    if not issues:
        return "(none)"
    return "\n".join(
        f"- {i.type}" + (f" in {i.kernel}" if i.kernel else "") + f": {i.evidence} ({i.source})"
        for i in issues
    )
