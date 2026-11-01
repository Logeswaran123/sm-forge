"""compute-sanitizer on one kernel file (parent side).

The program is ``smforge.harness.ncu_target``: a warm-up forward, which autotunes and so runs every
config, then one more. PyTorch's caching allocator is off, so an access past the end of a tensor
lands outside every allocation rather than in a neighbouring cached block, where memcheck would
not see it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from smforge.device.tools import find_tool
from smforge.models import SanitizerReport

TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
TIMEOUT = 1800
PRINT_LIMIT = 20
_PREFIX = "========= "
_SUMMARY = re.compile(r"(?:ERROR SUMMARY: |RACECHECK SUMMARY: .*\()(\d+) error")
_LOCATION = re.compile(r"^\s+at (\S+?)\+0x[0-9a-f]+(?: in (\S+))?")


class SanitizerError(RuntimeError):
    pass


class SanitizerUnavailable(SanitizerError):
    """compute-sanitizer fails on any program, not only on one kernel."""


def sanitize(
    kernel: Path,
    tool: str = "memcheck",
    spec: Path | None = None,
    variant: str | None = None,
    timeout: float = TIMEOUT,
) -> SanitizerReport:
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}; choose from {', '.join(TOOLS)}")
    binary = find_tool("compute-sanitizer")
    if binary is None:
        raise SanitizerError("compute-sanitizer not found")
    command = [
        binary,
        "--tool",
        tool,
        "--show-backtrace",
        "device",
        "--print-limit",
        str(PRINT_LIMIT),
        sys.executable,
        "-m",
        "smforge.harness.ncu_target",
        str(kernel),
    ]
    if spec is not None:
        command += ["--spec", str(spec)]
    if variant is not None:
        command += ["--variant", variant]
    env = {**os.environ, "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SanitizerError(str(e)) from e
    return parse(tool, out.stdout + out.stderr, out.returncode)


def parse(tool: str, text: str, returncode: int = 0) -> SanitizerReport:
    """The error count, and each distinct error with the kernel and line it points at."""
    lines = [line[len(_PREFIX) :] for line in text.splitlines() if line.startswith(_PREFIX)]
    counts = [int(m.group(1)) for line in lines if (m := _SUMMARY.search(line))]
    messages: list[str] = []
    for i, line in enumerate(lines):
        if not line or line[0].isspace() or "SUMMARY" in line or line == "COMPUTE-SANITIZER":
            continue
        if line.startswith(("Target application returned", "Error: process didn't terminate")):
            continue
        where = next((m for m in map(_LOCATION.match, lines[i + 1 : i + 3]) if m), None)
        if where:
            line += f" at {where.group(1)}" + (f" ({where.group(2)})" if where.group(2) else "")
        if line not in messages:
            messages.append(line)
    errors = counts[0] if counts else len(messages)
    crashed = returncode != 0 and not errors
    detail = None
    if crashed:
        tail = [line for line in text.strip().splitlines() if not line.startswith(_PREFIX)]
        detail = "\n".join(tail[-5:]) or f"exit code {returncode}"
    return SanitizerReport(
        tool=tool, ok=not errors and not crashed, errors=errors, messages=messages, detail=detail
    )
