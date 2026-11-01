"""Worker process: runs kernels on one GPU and answers JSON requests, one per line.

Requests arrive on stdin as ``{"id", "op", "args"}``. Replies go to the original stdout; anything
a kernel prints goes to stderr. A CUDA error that leaves the context unusable is reported as
fatal, and the process exits after its reply.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from smforge.config import TimingConfig
from smforge.corpus.kernelbench import verify
from smforge.corpus.measure import compiled, profile_reference
from smforge.device.noise import measure_noise
from smforge.device.probes import PROBES, run_probe
from smforge.device.query import query_device
from smforge.device.roofline import measure_roofline
from smforge.harness.correctness import ReferenceCache, check_kernel
from smforge.harness.cuda_facts import collect_cuda_facts
from smforge.harness.facts import collect_facts
from smforge.harness.launches import profile_launches, triton_names
from smforge.harness.timing import Timer, graph_callable
from smforge.harness.workload import Workload, clone, cuda_healthy, load_module
from smforge.models import FactsReport, Timing
from smforge.spec import load_spec


class Session:
    def __init__(self, device: int, cache_dir: Path):
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.device = torch.device("cuda", device)
        self.refs = ReferenceCache(cache_dir / "refs", self.device)
        self._timers: dict[str, Timer] = {}

    def timer(self, timing: dict) -> Timer:
        config = TimingConfig.model_validate(timing)
        key = config.model_dump_json()
        if key not in self._timers:
            self._timers.clear()
            self._timers[key] = Timer(config, self.device)
        return self._timers[key]

    def workload(
        self,
        spec: str | None,
        variant: str | None,
        index: int,
        fallback: str,
        dtype: str | None = None,
    ):
        return Workload.create(self.device, spec, variant, index, fallback, dtype)

    def runnable(self, kernel: str, workload: Workload, graph: bool = False):
        module = load_module(kernel)
        model = workload.build(module)
        inputs = clone(workload.inputs())

        def call():
            return model(*inputs)

        return (graph_callable(call) if graph else call), module

    def op_ping(self) -> dict:
        return {"device": torch.cuda.get_device_name(self.device), "pid": os.getpid()}

    def op_check(
        self,
        candidate: str,
        oracle: str,
        rtol: float,
        atol: float,
        draws: int,
        spec: str | None = None,
        variant: str | None = None,
        index: int = 0,
        dtype: str | None = None,
    ) -> dict:
        workload = self.workload(spec, variant, index, oracle, dtype)
        result = check_kernel(Path(candidate), Path(oracle), workload, rtol, atol, draws, self.refs)
        return result.model_dump(mode="json")

    def op_launches(
        self,
        kernel: str,
        spec: str | None = None,
        variant: str | None = None,
        index: int = 0,
        reps: int = 3,
    ) -> dict:
        fn, module = self.runnable(kernel, self.workload(spec, variant, index, kernel))
        return profile_launches(fn, triton_names(module), reps).model_dump(mode="json")

    def op_bench(
        self,
        kernel: str,
        timing: dict,
        spec: str | None = None,
        variant: str | None = None,
        index: int = 0,
        graph: bool = False,
    ) -> dict:
        workload = self.workload(spec, variant, index, kernel)
        fn, _ = self.runnable(kernel, workload, graph)
        result = self.timer(timing).time(fn)
        return _rates(result, workload, graph).model_dump(mode="json")

    def op_compare(
        self,
        baseline: str,
        candidate: str,
        timing: dict,
        spec: str | None = None,
        variant: str | None = None,
        index: int = 0,
        graph: bool = False,
    ) -> dict:
        workload = self.workload(spec, variant, index, baseline)
        baseline_fn, _ = self.runnable(baseline, workload, graph)
        candidate_fn, _ = self.runnable(candidate, workload, graph)
        result = self.timer(timing).compare(baseline_fn, candidate_fn)
        result.graph = graph
        return result.model_dump(mode="json")

    def op_facts(
        self,
        kernel: str,
        spec: str | None = None,
        variant: str | None = None,
        index: int = 0,
    ) -> dict:
        workload = self.workload(spec, variant, index, kernel)
        fn, _ = self.runnable(kernel, workload)
        fn()
        fn()
        sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        report = FactsReport(
            kernels=collect_facts(fn, sm_count) + collect_cuda_facts(fn, sm_count),
            flop=workload.flop,
            bytes_moved=workload.bytes_moved,
        )
        return report.model_dump(mode="json")

    def op_device(self) -> dict:
        return query_device(self.device.index).model_dump(mode="json")

    def op_probe_names(self) -> list[str]:
        return list(PROBES)

    def op_probe(self, name: str) -> dict:
        return run_probe(name, self.device).model_dump(mode="json")

    def op_roofline(self) -> dict:
        return measure_roofline(self.device).model_dump(mode="json")

    def op_noise(self, timing: dict) -> dict:
        return measure_noise(self.timer(timing), self.device).model_dump(mode="json")

    def op_verify_translation(self, reference: str, spec: str, variant: str) -> None:
        kernel_spec = load_spec(spec)
        verify(Path(reference), kernel_spec, kernel_spec.variant(variant).dims)

    def op_profile_reference(
        self, reference: str, spec: str, variant: str | None = None, index: int = 0
    ) -> dict:
        workload = self.workload(spec, variant, index, reference)
        return profile_reference(workload, load_module(reference)).model_dump(mode="json")

    def op_baseline(
        self,
        reference: str,
        mode: str,
        timing: dict,
        spec: str | None = None,
        variant: str | None = None,
        index: int = 0,
    ) -> dict:
        workload = self.workload(spec, variant, index, reference)
        model = compiled(workload.build(load_module(reference)), mode)
        inputs = clone(workload.inputs())
        result = self.timer(timing).time(lambda: model(*inputs))
        return _rates(result, workload, False).model_dump(mode="json")


def _rates(timing: Timing, workload: Workload, graph: bool) -> Timing:
    seconds = timing.time_us * 1e-6
    timing.graph = graph
    if workload.flop:
        timing.tflops = workload.flop / seconds / 1e12
    if workload.bytes_moved:
        timing.gbps = workload.bytes_moved / seconds / 1e9
    return timing


def serve(session: Session, requests, replies) -> None:
    for line in requests:
        request = json.loads(line)
        reply: dict[str, Any] = {"id": request["id"]}
        try:
            with torch.no_grad():
                handler = getattr(session, f"op_{request['op']}")
                reply.update(ok=True, result=handler(**request.get("args", {})))
        except Exception as e:
            reply.update(
                ok=False,
                error=f"{type(e).__name__}: {e}".strip(),
                traceback=traceback.format_exc(),
                fatal=getattr(e, "fatal", False) or not cuda_healthy(),
            )
        replies.write(json.dumps(reply) + "\n")
        replies.flush()
        if not reply["ok"] and reply["fatal"]:
            os._exit(3)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path(".smforge"))
    args = parser.parse_args(argv)
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    serve(Session(args.device, args.cache_dir), sys.stdin, replies)


if __name__ == "__main__":
    main()
