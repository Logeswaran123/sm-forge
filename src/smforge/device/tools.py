"""External NVIDIA tools: ``nvidia-smi`` queries, and checks that the profilers run here."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from smforge.models import ToolStatus

PROGRAM = "import torch; x = torch.ones(1 << 20, device='cuda'); print(float((x * 2).sum()))"
_TOOL_ERROR = re.compile(r"==ERROR==|=+ Error: |ERROR SUMMARY: [1-9]")


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    roots = [os.environ.get("CUDA_HOME"), "/usr/local/cuda"]
    roots += sorted((str(p) for p in Path("/usr/local").glob("cuda-*")), reverse=True)
    for root in filter(None, roots):
        path = Path(root) / "bin" / name
        if path.exists():
            return str(path)
    return None


def nvidia_smi(index: int, *fields: str) -> dict[str, str]:
    """Values of ``--query-gpu`` fields; fields the driver reports as N/A are left out."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return {}
    out = subprocess.run(
        [smi, "-i", str(index), f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        return {}
    values = [v.strip() for v in out.stdout.strip().split(",")]
    return {f: v for f, v in zip(fields, values, strict=False) if v and not v.startswith("[")}


def is_wsl() -> bool:
    version = Path("/proc/version")
    return version.exists() and "microsoft" in version.read_text().lower()


def check_tools(python: str = sys.executable, timeout: float = 600) -> list[ToolStatus]:
    """Run each profiler on a one-line CUDA program and report whether it worked."""
    with tempfile.TemporaryDirectory() as tmp:
        commands: dict[str, Callable[[str], list[str]]] = {
            "ncu": lambda tool: [tool, "--section", "SpeedOfLight", python, "-c", PROGRAM],
            "nsys": lambda tool: [
                tool,
                "profile",
                "--trace=cuda",
                "--force-overwrite=true",
                "-o",
                f"{tmp}/probe",
                python,
                "-c",
                PROGRAM,
            ],
            "compute-sanitizer": lambda tool: [tool, "--tool", "memcheck", python, "-c", PROGRAM],
        }
        return [_run_tool(name, command, timeout) for name, command in commands.items()]


def _run_tool(name: str, command: Callable[[str], list[str]], timeout: float) -> ToolStatus:
    path = find_tool(name)
    if path is None:
        return ToolStatus(name=name, ok=False, detail="not found")
    try:
        out = subprocess.run(command(path), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ToolStatus(name=name, path=path, ok=False, detail=f"timed out after {timeout:.0f}s")
    errors = [
        line.strip(" =")
        for line in (out.stdout + out.stderr).splitlines()
        if _TOOL_ERROR.search(line)
    ]
    if errors:
        return ToolStatus(name=name, path=path, ok=False, detail=errors[0])
    if out.returncode != 0:
        return ToolStatus(name=name, path=path, ok=False, detail=f"exit code {out.returncode}")
    return ToolStatus(name=name, path=path, ok=True)
