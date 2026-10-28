from pathlib import Path

import pytest
from test_seed import copy_problem, scripted

from smforge import render
from smforge.agents.optimize import optimize_stage, system_prompt
from smforge.config import Config, LLMConfig, TimingConfig
from smforge.harness.runner import Runner
from smforge.kb.loader import load_kb
from smforge.llm.client import LLMClient
from smforge.models import FactsReport, KernelFacts, Stage

KERNEL = Path(__file__).parent / "kernels" / "linear_relu" / "kernel.py"


def slow(kernel: str) -> str:
    return kernel.replace(
        "triton.cdiv(M, 64), triton.cdiv(N, 64)", "triton.cdiv(M, 16), triton.cdiv(N, 16)"
    ).replace(
        "BLOCK_M=64,\n            BLOCK_N=64,\n            BLOCK_K=32,",
        "BLOCK_M=16,\n            BLOCK_N=16,\n            BLOCK_K=16,",
    )


def test_system_prompt():
    text = system_prompt(Stage.AUTOTUNING, None, "## Constraints\n\n- rule", 0.02)
    assert "This stage is autotuning" in text
    assert "@triton.autotune" in text
    assert "faster by more than 2%" in text
    assert text.rstrip().endswith("single ```python block.")


@pytest.mark.gpu
def test_optimize_stage(monkeypatch, tmp_path):
    problem = copy_problem(tmp_path)
    good = KERNEL.read_text()
    assert slow(good) != good
    problem.seed.write_text(slow(good))
    changed = good.replace(
        "def __init__(self, in_features, out_features):",
        "def __init__(self, in_features, out_features, scale=1.0):",
    )
    config = Config(cache_dir=tmp_path / "cache", timing=TimingConfig(warmup=3, iters=40, rounds=4))
    llm = LLMClient(LLMConfig())
    replies = ["no code", f"```python\n{changed}```", f"```python\n{slow(good)}```"]
    requests = scripted(monkeypatch, llm, [*replies, f"```python\n{good}```"])
    with Runner(config) as runner:
        result = optimize_stage(
            problem, Stage.AUTOTUNING, problem.seed, runner, llm, load_kb(), None, 4, tmp_path / "w"
        )
    outcomes = [a.outcome for a in result.attempts]
    assert outcomes[:2] == ["no_code", "interface"]
    assert outcomes[2] in ("parity", "slower")
    assert outcomes[3] == "accepted"
    assert result.accepted and result.speedup > 1.5
    assert result.kernel.read_text() == good
    assert (tmp_path / "w" / "output.py").read_text() == good
    assert "Kernel changes from the current kernel" in requests[3]["messages"][-1]["content"]
    assert len(requests[3]["messages"]) == 5
    system = requests[0]["system"][0]["text"]
    assert "## Constraints" in system and "autotune_small_config_space" in system
    assert "Current kernel (`kernel.py`)" in requests[0]["messages"][0]["content"]


def test_fact_deltas():
    def kernel(name, regs, spills, warps=4):
        return KernelFacts(
            name=name,
            grid=[64, 1, 1],
            ctas=64,
            num_warps=warps,
            num_stages=3,
            n_regs=regs,
            n_spills=spills,
            shared_bytes=24576,
        )

    before = FactsReport(kernels=[kernel("gemm", 96, 0), kernel("bias_relu", 32, 0)])
    after = FactsReport(kernels=[kernel("gemm", 168, 24, warps=8), kernel("pool", 40, 0)])
    assert render.fact_deltas(before, after).splitlines() == [
        "gemm: regs 96 -> 168, spills 0 -> 24, warps 4 -> 8",
        "pool: new; regs 40, spills 0, 4 warps, 64 CTAs",
        "bias_relu: removed",
    ]
    assert render.fact_deltas(before, before).splitlines()[0] == "gemm: unchanged"
