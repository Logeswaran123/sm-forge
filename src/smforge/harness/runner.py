"""Parent side of the harness: drives a worker subprocess and restarts it when it dies.

A kernel that faults leaves the CUDA context unusable for the rest of the process, so no kernel
ever runs in the caller's process.
"""

from __future__ import annotations

import itertools
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path
from typing import Any

from smforge.config import Config, get_config
from smforge.harness.source import KernelSource
from smforge.harness.tolerance import resolve_tolerance
from smforge.models import (
    CheckResult,
    Comparison,
    Dsl,
    FactsReport,
    LaunchProfile,
    ReferenceProfile,
    Timing,
)
from smforge.spec import load_spec

STDERR_TAIL = 4000


class WorkerError(RuntimeError):
    def __init__(self, message: str, *, detail: str = "", fatal: bool = False):
        super().__init__(message)
        self.detail = detail
        self.fatal = fatal


class Runner:
    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._proc: subprocess.Popen | None = None
        self._log: Path | None = None
        self._ids = itertools.count()
        self._starts = itertools.count()

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        log_dir = self.config.cache_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = log_dir / f"worker-{os.getpid()}-{next(self._starts)}.log"
        with open(self._log, "wb") as log:
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "smforge.harness.worker",
                    "--device",
                    str(self.config.device),
                    "--cache-dir",
                    str(self.config.cache_dir),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
            )

    def close(self) -> None:
        if self._proc is None:
            return
        self._proc.stdin.close()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def call(self, op: str, *, timeout: float | None = None, **args: Any) -> Any:
        if self._proc is None or self._proc.poll() is not None:
            self.start()
        request_id = next(self._ids)
        self._proc.stdin.write(json.dumps({"id": request_id, "op": op, "args": args}) + "\n")
        self._proc.stdin.flush()
        reply = json.loads(self._readline(timeout or self.config.worker_timeout))
        if reply["ok"]:
            return reply["result"]
        if reply["fatal"]:
            self._proc.wait()
            self._proc = None
        raise WorkerError(reply["error"], detail=reply["traceback"], fatal=reply["fatal"])

    def _readline(self, timeout: float) -> str:
        with selectors.DefaultSelector() as selector:
            selector.register(self._proc.stdout, selectors.EVENT_READ)
            ready = selector.select(timeout)
        if not ready:
            self._proc.kill()
            self._proc.wait()
            self._proc = None
            raise WorkerError(f"timed out after {timeout:.0f}s", detail=self.stderr(), fatal=True)
        line = self._proc.stdout.readline()
        if not line:
            code = self._proc.wait()
            self._proc = None
            raise WorkerError(f"worker exited with code {code}", detail=self.stderr(), fatal=True)
        return line

    def stderr(self) -> str:
        """The end of the current worker's stderr."""
        if self._log is None or not self._log.exists():
            return ""
        return self._log.read_text(errors="replace")[-STDERR_TAIL:]

    def check(
        self,
        candidate: Path,
        oracle: Path,
        *,
        spec: Path | None = None,
        variant: str | None = None,
        index: int = 0,
        rtol: float | None = None,
        atol: float | None = None,
        draws: int | None = None,
        dtype: str | None = None,
    ) -> CheckResult:
        """``dtype`` replaces the variant's dtype, and its tolerance with that dtype's default."""
        variant, resolved, variant_spec = _resolve(spec, variant, index)
        if dtype is not None and dtype != resolved:
            resolved, variant_spec = dtype, None
        else:
            dtype = None
        rtol, atol = resolve_tolerance(resolved, variant_spec, self.config.correctness, rtol, atol)
        try:
            result = self.call(
                "check",
                candidate=str(candidate),
                oracle=str(oracle),
                rtol=rtol,
                atol=atol,
                draws=draws or self.config.correctness.draws,
                dtype=dtype,
                **_workload(spec, variant, index),
            )
        except WorkerError as e:
            error = f"{e}\n{_tail(e.detail)}" if e.fatal and e.detail else str(e)
            return CheckResult(passed=False, rtol=rtol, atol=atol, error=error)
        return CheckResult.model_validate(result)

    def launches(
        self, kernel: Path, *, spec: Path | None = None, variant: str | None = None, index: int = 0
    ) -> LaunchProfile:
        """Profile in a fresh worker; on WSL2 the profiler goes quiet in an older one.

        An empty profile ends that worker, and the profile is retried once in a new one.
        """
        variant = _resolve(spec, variant, index)[0]
        args = {"kernel": str(kernel), **_workload(spec, variant, index)}
        self.close()
        try:
            result = self.call("launches", **args)
        except WorkerError as e:
            if not str(e).startswith("ProfilerUnavailable"):
                raise
            result = self.call("launches", **args)
        return LaunchProfile.model_validate(result)

    def bench(
        self,
        kernel: Path,
        *,
        spec: Path | None = None,
        variant: str | None = None,
        index: int = 0,
        graph: bool = False,
    ) -> Timing:
        variant = _resolve(spec, variant, index)[0]
        result = self.call(
            "bench",
            kernel=str(kernel),
            timing=self.config.timing.model_dump(),
            graph=graph,
            **_workload(spec, variant, index),
        )
        return Timing.model_validate(result)

    def compare(
        self,
        baseline: Path,
        candidate: Path,
        *,
        spec: Path | None = None,
        variant: str | None = None,
        index: int = 0,
        graph: bool = False,
    ) -> Comparison:
        variant = _resolve(spec, variant, index)[0]
        result = self.call(
            "compare",
            baseline=str(baseline),
            candidate=str(candidate),
            timing=self.config.timing.model_dump(),
            graph=graph,
            **_workload(spec, variant, index),
        )
        return Comparison.model_validate(result)

    def facts(
        self, kernel: Path, *, spec: Path | None = None, variant: str | None = None, index: int = 0
    ) -> FactsReport:
        """A ``cuda`` file's facts come from the profiler, so like ``launches`` they are taken in a
        fresh worker and retried once in a new one."""
        variant = _resolve(spec, variant, index)[0]
        args = {"kernel": str(kernel), **_workload(spec, variant, index)}
        if KernelSource(kernel.read_text()).dsl is not Dsl.CUDA:
            return FactsReport.model_validate(self.call("facts", **args))
        self.close()
        try:
            result = self.call("facts", **args)
        except WorkerError as e:
            if not str(e).startswith("ProfilerUnavailable"):
                raise
            result = self.call("facts", **args)
        return FactsReport.model_validate(result)

    def profile_reference(
        self, reference: Path, *, spec: Path, variant: str | None = None, index: int = 0
    ) -> ReferenceProfile:
        variant = _resolve(spec, variant, index)[0]
        result = self.call(
            "profile_reference", reference=str(reference), **_workload(spec, variant, index)
        )
        return ReferenceProfile.model_validate(result)

    def baseline(
        self,
        reference: Path,
        mode: str,
        *,
        spec: Path | None = None,
        variant: str | None = None,
        index: int = 0,
        timeout: float | None = None,
    ) -> Timing:
        variant = _resolve(spec, variant, index)[0]
        result = self.call(
            "baseline",
            timeout=timeout,
            reference=str(reference),
            mode=mode,
            timing=self.config.timing.model_dump(),
            **_workload(spec, variant, index),
        )
        return Timing.model_validate(result)


def _resolve(spec: Path | None, variant: str | None, index: int):
    """The variant name, its dtype and its entry, resolved before anything reaches the worker."""
    if spec is None:
        return variant, "float32", None
    kernel_spec = load_spec(spec)
    entry = kernel_spec.variant(variant, index)
    return kernel_spec.resolve(variant), kernel_spec.dtype(entry), entry


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _workload(spec: Path | None, variant: str | None, index: int) -> dict:
    return {"spec": str(spec) if spec else None, "variant": variant, "index": index}
