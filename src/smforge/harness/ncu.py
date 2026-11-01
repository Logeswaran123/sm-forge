"""Nsight Compute on one forward of a kernel file (parent side): counters, and section profiles.

ncu runs ``smforge.harness.ncu_target`` in a new process and profiles only the forward inside the
profiler range, so compilation and autotuning launches are left out. Loads that go through
cp.async or TMA do not count as LSU global loads, so ``load_bytes_per_sector`` is ``None`` for a
kernel without plain global loads.
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from smforge.device.tools import find_tool
from smforge.models import NcuKernel, NcuMetric, NcuProfile, NcuRule

METRICS = {
    "duration": "gpu__time_duration.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "load": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
    "store": "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    "conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "wavefronts": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
}
TIMEOUT = 900
SECTIONS = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "LaunchStats",
    "WarpStateStats",
    "SourceCounters",
)
MAX_LAUNCHES = 16
KEPT = {
    "GPU Speed Of Light Throughput": (
        "Duration",
        "Compute (SM) Throughput",
        "Memory Throughput",
        "DRAM Throughput",
        "L1/TEX Cache Throughput",
        "L2 Cache Throughput",
    ),
    "Memory Workload Analysis": ("Memory Throughput", "Mem Busy", "L1/TEX Hit Rate", "L2 Hit Rate"),
    "Launch Statistics": (
        "Registers Per Thread",
        "Dynamic Shared Memory Per Block",
        "Static Shared Memory Per Block",
        "Waves Per SM",
    ),
    "Occupancy": (
        "Theoretical Occupancy",
        "Achieved Occupancy",
        "Block Limit Registers",
        "Block Limit Shared Mem",
        "Block Limit Warps",
    ),
    "Warp State Statistics": (
        "Warp Cycles Per Issued Instruction",
        "Avg. Active Threads Per Warp",
    ),
    "Source Counters": ("Branch Efficiency", "Avg. Divergent Branches"),
}
_SECTION_LABELS = {
    "GPU Speed Of Light Throughput": "speed of light",
    "Memory Workload Analysis": "memory",
    "Launch Statistics": "launch",
    "Occupancy": "occupancy",
    "Warp State Statistics": "warps",
    "Source Counters": "branches",
}
_SHORT = {
    "Duration": "duration",
    "Compute (SM) Throughput": "SM",
    "Memory Throughput": "memory",
    "DRAM Throughput": "DRAM",
    "L1/TEX Cache Throughput": "L1",
    "L2 Cache Throughput": "L2",
    "L1/TEX Hit Rate": "L1 hit",
    "L2 Hit Rate": "L2 hit",
    "Registers Per Thread": "registers",
    "Dynamic Shared Memory Per Block": "dynamic smem",
    "Static Shared Memory Per Block": "static smem",
    "Waves Per SM": "waves",
    "Theoretical Occupancy": "theoretical",
    "Achieved Occupancy": "achieved",
    "Block Limit Registers": "block limit by registers",
    "Block Limit Shared Mem": "block limit by smem",
    "Block Limit Warps": "block limit by warps",
    "Warp Cycles Per Issued Instruction": "cycles per issued instruction",
    "Avg. Active Threads Per Warp": "active threads per warp",
    "Mem Busy": "busy",
    "Branch Efficiency": "efficiency",
    "Avg. Divergent Branches": "divergent",
    ("Memory Workload Analysis", "Memory Throughput"): "throughput",
}
_PLAIN_UNITS = {"register/thread", "block", "cycle", "warp", "thread"}
_SCALE = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
_UNIT = re.compile(r"^(n|u|m|K|M|G|T)?(s|second|byte|byte/s|byte/second|byte/block|bit|Hz|hz)$")
MIN_CHANGE = 0.01


class NcuError(RuntimeError):
    pass


class NcuUnavailable(NcuError):
    """ncu fails on any program, not only on one kernel: the counters are off, or it is gone."""


def profile_ncu(
    kernel: Path, names: set[str], spec: Path | None = None, variant: str | None = None
) -> list[NcuKernel]:
    """Counters for the launches of the kernels named ``names``, one entry per name."""
    options = ["--page", "raw", "--metrics", ",".join(METRICS.values())]
    return parse(_run(kernel, names, options, spec, variant))


def profile_sections(
    kernel: Path,
    names: set[str],
    spec: Path | None = None,
    variant: str | None = None,
    launches: int = MAX_LAUNCHES,
) -> list[NcuProfile]:
    """The SECTIONS for the first launch of each kernel in ``names``, at most ``launches``."""
    options = ["--page", "details", "--launch-count", str(launches)]
    for section in SECTIONS:
        options += ["--section", section]
    return parse_details(_run(kernel, names, options, spec, variant))


def _run(
    kernel: Path, names: set[str], options: list[str], spec: Path | None, variant: str | None
) -> str:
    binary = find_tool("ncu")
    if binary is None:
        raise NcuError("ncu not found")
    command = [
        binary,
        "--csv",
        "--profile-from-start",
        "off",
        *options,
        "--kernel-name",
        "regex:^(" + "|".join(re.escape(name) for name in sorted(names)) + ")$",
        sys.executable,
        "-m",
        "smforge.harness.ncu_target",
        str(kernel),
    ]
    if spec is not None:
        command += ["--spec", str(spec)]
    if variant is not None:
        command += ["--variant", variant]
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise NcuError(str(e)) from e
    if out.returncode != 0 or "ERR_" in out.stdout:
        detail = (out.stderr or out.stdout).strip().splitlines()
        raise NcuError(detail[-1] if detail else f"ncu exited with {out.returncode}")
    return out.stdout


def parse_details(text: str) -> list[NcuProfile]:
    """One profile per kernel name, from its first launch in ``--page details`` CSV."""
    table = "\n".join(line for line in text.splitlines() if line.startswith('"'))
    profiles: dict[str, NcuProfile] = {}
    first: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(table)):
        name = row["Kernel Name"]
        if first.setdefault(name, row["ID"]) != row["ID"]:
            continue
        profile = profiles.setdefault(
            name, NcuProfile(kernel=name, grid=row["Grid Size"], block=row["Block Size"])
        )
        if row.get("Rule Name"):
            speedup = row.get("Estimated Speedup") or ""
            profile.rules.append(
                NcuRule(
                    section=row["Section Name"],
                    name=row["Rule Name"],
                    kind=row.get("Rule Type") or "",
                    scope=row.get("Estimated Speedup Type") or "",
                    speedup=float(speedup) if speedup else None,
                    description=row.get("Rule Description") or "",
                )
            )
        elif row["Metric Name"]:
            profile.metrics.append(
                NcuMetric(
                    section=row["Section Name"],
                    name=row["Metric Name"],
                    unit=row["Metric Unit"],
                    value=row["Metric Value"],
                )
            )
    return list(profiles.values())


def render_profile(profile: NcuProfile, rules: int = 4) -> str:
    """About 30 lines: the kept metrics of each section, then ncu's largest estimated gains."""
    lines = [f"{profile.kernel} (grid {profile.grid}, block {profile.block}):"]
    values = profile.values()
    for section, names in KEPT.items():
        parts = [
            f"{_short(section, name)} {_value(*values[section, name])}"
            for name in names
            if (section, name) in values
        ]
        if parts:
            lines.append(f"  {_SECTION_LABELS[section]}: " + ", ".join(parts))
    suggestions = sorted(
        (r for r in profile.rules if r.kind == "OPT" and r.scope == "global" and r.speedup),
        key=lambda r: -r.speedup,
    )[:rules]
    if suggestions:
        lines.append("  ncu estimates for the whole kernel:")
    for rule in suggestions:
        lines.append(
            f"    - {rule.name}, up to {rule.speedup:.0f}%: {_first_sentence(rule.description)}"
        )
    return "\n".join(lines)


def diff_profiles(before: list[NcuProfile], after: list[NcuProfile]) -> str:
    """Kept metrics that changed by ``MIN_CHANGE`` or more, kernel by kernel and section by
    section; kernels pair by name, else by order."""
    by_name = {p.kernel: p for p in before}
    lines = []
    for i, new in enumerate(after):
        old = by_name.get(new.kernel) or (before[i] if i < len(before) else None)
        if old is None:
            lines.append(f"{new.kernel}: new kernel")
            continue
        label = new.kernel if old.kernel == new.kernel else f"{old.kernel} -> {new.kernel}"
        old_values, new_values = old.values(), new.values()
        changes = []
        for section, names in KEPT.items():
            rows = []
            for name in names:
                key = (section, name)
                if key not in old_values or key not in new_values:
                    continue
                a, b = old_values[key], new_values[key]
                if _moved(a, b):
                    rows.append(
                        f"    {_short(section, name)}: {_value(*a)} -> {_value(*b)}{_change(a, b)}"
                    )
            if rows:
                changes += [f"  {_SECTION_LABELS[section]}:", *rows]
        old_rules, new_rules = _hints(old), _hints(new)
        if gone := sorted(old_rules - new_rules):
            changes.append(f"  ncu estimates gone: {', '.join(gone)}")
        if added := sorted(new_rules - old_rules):
            changes.append(f"  ncu estimates new: {', '.join(added)}")
        lines.append(f"{label}:" + ("\n" + "\n".join(changes) if changes else " no change"))
    return "\n".join(lines)


def _hints(profile: NcuProfile) -> set[str]:
    return {r.name for r in profile.rules if r.kind == "OPT" and r.scope == "global" and r.speedup}


def _short(section: str, name: str) -> str:
    return _SHORT.get((section, name)) or _SHORT.get(name) or name[0].lower() + name[1:]


def _value(value: str, unit: str) -> str:
    """Times in us and bandwidths in GB/s, whatever prefix ncu chose."""
    if unit == "%":
        return f"{value}%"
    if unit in _PLAIN_UNITS:
        return value
    match = _UNIT.match(unit)
    if match and match.group(2) in ("s", "second"):
        return f"{_base(value, unit) * 1e6:.1f} us"
    if match and match.group(2) in ("byte/s", "byte/second"):
        return f"{_base(value, unit) / 1e9:.1f} GB/s"
    return f"{value} {unit}".strip()


def _base(value: str, unit: str) -> float:
    """``value`` in the unit without its prefix: 1.42 ms and 1420 us compare equal."""
    number = _number(value)
    match = _UNIT.match(unit)
    return number * _SCALE[match.group(1) or ""] if match else number


def _moved(a: tuple[str, str], b: tuple[str, str]) -> bool:
    try:
        x, y = _base(*a), _base(*b)
    except ValueError:
        return a != b
    return abs(y - x) >= MIN_CHANGE * abs(x) if x else y != 0


def _change(a: tuple[str, str], b: tuple[str, str]) -> str:
    try:
        x, y = _base(*a), _base(*b)
    except ValueError:
        return ""
    return f" ({(y - x) / x:+.0%})" if x else ""


def _first_sentence(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    end = re.search(r"\.(?: |[A-Z])", text)
    text = text if end is None else text[: end.start() + 1]
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse(text: str) -> list[NcuKernel]:
    table = "\n".join(line for line in text.splitlines() if line.startswith('"'))
    rows = list(csv.DictReader(io.StringIO(table)))
    launches: dict[str, list[dict]] = defaultdict(list)
    for row in rows[1:]:
        launches[row["Kernel Name"]].append(row)
    return [_kernel(name, runs) for name, runs in launches.items()]


def _kernel(name: str, runs: list[dict]) -> NcuKernel:
    def mean(key: str) -> float:
        return sum(_number(run[METRICS[key]]) for run in runs) / len(runs)

    load, store = mean("load"), mean("store")
    wavefronts = mean("wavefronts")
    return NcuKernel(
        name=name,
        launches=len(runs),
        duration_us=mean("duration") / 1e3,
        sm_pct=mean("sm_pct"),
        dram_pct=mean("dram_pct"),
        load_bytes_per_sector=load or None,
        store_bytes_per_sector=store or None,
        shared_conflicts_per_wavefront=mean("conflicts") / wavefronts if wavefronts else None,
    )


def _number(value: str) -> float:
    return float(value.replace(",", "")) if value else 0.0
