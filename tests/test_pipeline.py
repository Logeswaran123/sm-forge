import json
from pathlib import Path

import pytest
from test_optimize import slow
from test_seed import copy_problem, message

from smforge.agents import analyze, pipeline
from smforge.agents.analyze import Measurements, measure, pre_analyze
from smforge.agents.optimize import StageResult
from smforge.agents.pipeline import PipelineResult, finalize, memcheck_summary, run_pipeline
from smforge.agents.planner import ROUTES, order, plan, routed
from smforge.config import Config, LLMConfig, TimingConfig
from smforge.device.tools import tool_status
from smforge.harness.ncu import NcuError, NcuUnavailable
from smforge.harness.runner import Runner
from smforge.kb.loader import load_kb
from smforge.llm.client import LLMClient
from smforge.models import (
    Comparison,
    DeviceInfo,
    DeviceProfile,
    FactsReport,
    Issue,
    IssueType,
    KernelFacts,
    Launch,
    LaunchCategory,
    LaunchProfile,
    NcuKernel,
    SanitizerReport,
    Stage,
    Timing,
    ToolStatus,
    Verdict,
)

KERNEL = Path(__file__).parent / "kernels" / "linear_relu" / "kernel.py"


def test_every_issue_routes_to_a_stage():
    assert set(ROUTES) == set(IssueType)
    assert set(ROUTES.values()) == set(Stage)


def test_order_keeps_hard_dependencies():
    stages = [Stage.AUTOTUNING, Stage.FUSION, Stage.MEMORY_ACCESS, Stage.ALGORITHMIC]
    assert order(stages) == [
        Stage.MEMORY_ACCESS,
        Stage.ALGORITHMIC,
        Stage.FUSION,
        Stage.AUTOTUNING,
    ]
    assert order([Stage.DEVICE_SPECIFIC, Stage.DTYPE_FIX]) == [
        Stage.DEVICE_SPECIFIC,
        Stage.DTYPE_FIX,
    ]


def issue(kind: IssueType, source: str = "facts") -> Issue:
    return Issue(type=kind, evidence="e", source=source)


def test_plan(monkeypatch):
    issues = [
        issue(IssueType.UNTUNED_CONFIG),
        issue(IssueType.REGISTER_SPILL),
        issue(IssueType.UNFUSED_OPS, "llm"),
    ]
    assert routed(issues) == [Stage.FUSION, Stage.DEVICE_SPECIFIC, Stage.AUTOTUNING]
    llm = LLMClient(LLMConfig())
    reply = {"stages": ["autotuning", "fusion"], "reason": "fusion first"}
    monkeypatch.setattr(llm, "_send", lambda request: message(json.dumps(reply)))
    stages, reason, usage = plan(llm, "summary", issues)
    assert stages == [Stage.FUSION, Stage.DEVICE_SPECIFIC, Stage.AUTOTUNING]
    assert reason == "fusion first" and usage.calls == 1
    stages, _, _ = plan(llm, "summary", issues, max_stages=1)
    assert stages == [Stage.FUSION, Stage.AUTOTUNING]

    def fail(request):
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "_send", fail)
    stages, reason, _ = plan(llm, "summary", issues)
    assert stages == routed(issues) and reason.startswith("fallback")


def facts(**kw) -> KernelFacts:
    base = {
        "name": "matmul_kernel",
        "grid": [40, 1, 1],
        "ctas": 40,
        "num_warps": 4,
        "num_stages": 1,
        "n_regs": 255,
        "n_spills": 12,
        "shared_bytes": 8192,
        "ctas_per_sm": 1,
        "waves": 0.48,
        "mma": {},
        "instructions": {"FFMA": 1000},
    }
    return KernelFacts(**{**base, **kw})


CODE = """
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr):
    acc = tl.dot(a, b, acc)


class Model:
    def forward(self, x):
        out = x.item()
        return out
"""


def test_pre_analyze(monkeypatch):
    profile = DeviceProfile(
        device=DeviceInfo.model_validate(
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 5080",
                "capability": [12, 0],
                "sm_count": 84,
                "memory_bytes": 1,
                "l2_bytes": 1,
                "smem_per_block": 101376,
                "smem_per_sm": 102400,
                "regs_per_sm": 65536,
                "max_threads_per_sm": 1536,
                "warp_size": 32,
                "memory_bus_bits": 256,
                "torch": "2.11.0",
                "triton": "3.6.0",
            }
        )
    )
    launches = LaunchProfile(
        launches=[
            Launch(name="matmul_kernel", category=LaunchCategory.TRITON, calls=5, duration_us=20)
        ],
        kernel_us=20.0,
    )
    ncu = [
        NcuKernel(
            name="matmul_kernel",
            launches=1,
            duration_us=10,
            sm_pct=20,
            dram_pct=60,
            load_bytes_per_sector=8.0,
            shared_conflicts_per_wavefront=0.5,
        )
    ]
    timing = Timing(
        time_us=1.0,
        quantile=0.2,
        median_us=1.0,
        mean_us=1.0,
        min_us=1.0,
        max_us=1.0,
        samples=1,
        graph=False,
    )
    m = Measurements(timing, FactsReport(kernels=[facts()]), launches, ncu)
    found = {(i.type, i.source) for i in pre_analyze(CODE, m, "float32", profile)}
    assert found == {
        (IssueType.REGISTER_SPILL, "facts"),
        (IssueType.NO_TENSOR_CORE_MMA, "facts"),
        (IssueType.LOW_OCCUPANCY, "facts"),
        (IssueType.GRID_UNDERFILL, "facts"),
        (IssueType.TF32_PRECISION_RISK, "source"),
        (IssueType.UNTUNED_CONFIG, "source"),
        (IssueType.HOST_SYNC_IN_FORWARD, "source"),
        (IssueType.LAUNCH_BOUND, "facts"),
        (IssueType.UNCOALESCED_ACCESS, "ncu"),
        (IssueType.BANK_CONFLICT, "ncu"),
    }
    healthy = facts(
        n_spills=0, mma={"HMMA": 64}, num_stages=3, ctas=4096, ctas_per_sm=4, waves=12.2
    )
    m = Measurements(
        timing, FactsReport(kernels=[healthy]), launches.model_copy(update={"launches": []}), None
    )
    tuned = CODE.replace("@triton.jit", "@triton.autotune(configs=[], key=[])\n@triton.jit")
    tuned = tuned.replace("out = x.item()", "out = x").replace(
        "tl.dot(a, b, acc)", "tl.dot(a, b, acc, input_precision='ieee')"
    )
    assert pre_analyze(tuned, m, "float32", profile) == []


class FakeRunner:
    def facts(self, kernel, **args):
        return FactsReport(kernels=[facts()])

    def bench(self, kernel, **args):
        return Timing(
            time_us=10, quantile=0.2, median_us=10, mean_us=10, min_us=10, max_us=10, samples=1
        )

    def launches(self, kernel, **args):
        return LaunchProfile(launches=[], kernel_us=10)


def test_measure_stops_when_ncu_breaks(monkeypatch, tmp_path):
    problem = copy_problem(tmp_path)

    def fail(*args, **kw):
        raise NcuError("The application returned an error code (9).")

    monkeypatch.setattr(analyze, "profile_ncu", fail)
    monkeypatch.setattr(analyze, "tool_status", lambda name: ToolStatus(name=name, ok=True))
    assert measure(FakeRunner(), problem.seed, problem).ncu is None
    monkeypatch.setattr(
        analyze,
        "tool_status",
        lambda name: ToolStatus(name=name, ok=False, detail="ERR_NVGPUCTRPERM"),
    )
    with pytest.raises(NcuUnavailable, match="ERR_NVGPUCTRPERM"):
        measure(FakeRunner(), problem.seed, problem)
    assert measure(FakeRunner(), problem.seed, problem, ncu=False).ncu is None


def test_finalize_skips_what_memcheck_flags(monkeypatch, tmp_path):
    problem = copy_problem(tmp_path)
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    stages = [
        StageResult(problem.id, Stage.AUTOTUNING, True, first, 20, 10),
        StageResult(problem.id, Stage.FUSION, False, first, 10, 10),
        StageResult(problem.id, Stage.MEMORY_ACCESS, True, second, 10, 5),
    ]
    result = PipelineResult(problem.id, "", [], "", [], stages, 20, 20, problem.seed)
    flagged = SanitizerReport(tool="memcheck", ok=False, errors=3, messages=["Invalid write"])
    monkeypatch.setattr(
        pipeline,
        "sanitize",
        lambda kernel, *a: (
            flagged if kernel == second else SanitizerReport(tool="memcheck", ok=True, errors=0)
        ),
    )

    class Compare:
        def compare(self, baseline, candidate, **kw):
            return Comparison(
                baseline_us=20,
                candidate_us=10,
                speedup=2,
                verdict=Verdict.FASTER,
                min_effect=0.02,
                quantile=0.2,
                ratios=[2.0],
            )

    finalize(result, problem, Compare(), memcheck=True)
    assert result.kernel == first and result.final_us == 10
    assert memcheck_summary(result) == "memory_access=3 errors,autotuning=ok"
    result.kernel, result.memcheck = problem.seed, []
    finalize(result, problem, Compare(), memcheck=False)
    assert result.kernel == second and result.memcheck == []


@pytest.mark.gpu
def test_run_pipeline(monkeypatch, tmp_path):
    problem = copy_problem(tmp_path)
    good = KERNEL.read_text()
    problem.seed.write_text(slow(good))
    llm = LLMClient(LLMConfig())
    requests = []

    def send(request):
        requests.append(request)
        schema = request["output_config"].get("format", {}).get("schema", {})
        if "summary" in schema.get("properties", {}):
            return message(json.dumps({"summary": "tiles are too small", "issues": []}))
        if "stages" in schema.get("properties", {}):
            return message(json.dumps({"stages": ["autotuning"], "reason": "tiles"}))
        return message(f"```python\n{good}```")

    monkeypatch.setattr(llm, "_send", send)
    config = Config(cache_dir=tmp_path / "cache", timing=TimingConfig(warmup=3, iters=40, rounds=4))
    with Runner(config) as runner:
        memcheck = tool_status("compute-sanitizer").ok
        result = run_pipeline(
            problem, runner, llm, load_kb(), None, 2, tmp_path / "w", ncu=False, memcheck=memcheck
        )
    assert Stage.AUTOTUNING in result.plan
    assert any(s.accepted for s in result.stages)
    assert result.kernel.read_text() == good and result.speedup > 1.5
    assert "untuned_config" in requests[0]["messages"][0]["content"]
    record = json.loads((tmp_path / "w" / "pipeline.json").read_text())
    assert record["summary"] == "tiles are too small"
    assert [m["ok"] for m in record["memcheck"]] == ([True] if memcheck else [])
    assert (tmp_path / "w" / "output.py").read_text() == good
