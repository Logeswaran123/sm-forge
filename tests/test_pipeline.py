import json
from pathlib import Path

import pytest
from test_optimize import slow
from test_seed import copy_problem, message

from smforge.agents.analyze import Measurements, pre_analyze
from smforge.agents.pipeline import run_pipeline
from smforge.agents.planner import ROUTES, order, plan, routed
from smforge.config import Config, LLMConfig, TimingConfig
from smforge.harness.runner import Runner
from smforge.kb.loader import load_kb
from smforge.llm.client import LLMClient
from smforge.models import (
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
    Stage,
    Timing,
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
        result = run_pipeline(problem, runner, llm, load_kb(), None, 2, tmp_path / "w")
    assert Stage.AUTOTUNING in result.plan
    assert any(s.accepted for s in result.stages)
    assert result.kernel.read_text() == good and result.speedup > 1.5
    assert "untuned_config" in requests[0]["messages"][0]["content"]
    record = json.loads((tmp_path / "w" / "pipeline.json").read_text())
    assert record["summary"] == "tiles are too small"
    assert (tmp_path / "w" / "output.py").read_text() == good
