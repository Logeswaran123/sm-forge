"""Nsight Compute counters for one forward of a kernel file (parent side).

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
from smforge.models import NcuKernel

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


class NcuError(RuntimeError):
    pass


def profile_ncu(
    kernel: Path, names: set[str], spec: Path | None = None, variant: str | None = None
) -> list[NcuKernel]:
    """Counters for the launches of the kernels named ``names``, one entry per name."""
    binary = find_tool("ncu")
    if binary is None:
        raise NcuError("ncu not found")
    command = [
        binary,
        "--csv",
        "--page",
        "raw",
        "--profile-from-start",
        "off",
        "--metrics",
        ",".join(METRICS.values()),
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
    out = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
    if out.returncode != 0 or "ERR_" in out.stdout:
        detail = (out.stderr or out.stdout).strip().splitlines()
        raise NcuError(detail[-1] if detail else f"ncu exited with {out.returncode}")
    return parse(out.stdout)


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
