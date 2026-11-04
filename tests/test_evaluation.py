import json
from pathlib import Path

import pytest
from test_optimize import slow
from test_seed import copy_problem, message

from smforge.config import Config, LLMConfig, TimingConfig
from smforge.evaluation import (
    LLMUse,
    ProblemEval,
    RunConfig,
    differences,
    evaluate,
    load_run,
    report,
    roofline_us,
    save,
    summarize,
)
from smforge.harness.runner import Runner
from smforge.kb.loader import load_kb
from smforge.llm.client import LLMClient
from smforge.models import DeviceInfo, DeviceProfile, Roofline

KERNEL = Path(__file__).parent / "kernels" / "linear_relu" / "kernel.py"
CONFIG = RunConfig(
    split="eval", dsl="triton", model="m", effort="medium", max_iters=2, max_stages=2
)


def problem_eval(name: str, seed: float, final: float, **kw) -> ProblemEval:
    return ProblemEval(
        problem=name,
        family="gemm",
        seed_us=seed,
        final_us=final,
        kept_seed=seed == final,
        baselines_us={"eager": 2 * seed, "compile": seed},
        llm=LLMUse(calls=4, input_tokens=100, cache_read_tokens=300, cost_usd=1.0),
        wall_s=120,
        **kw,
    )


def test_summarize():
    problems = [
        problem_eval("a", 100, 50, seed_kernel_us=90, final_kernel_us=30, roofline_us=60),
        problem_eval("d", 100, 100, roofline_us=105),
        problem_eval("b", 100, 100),
        problem_eval("c", 100, 125, correctness={"kernelbench": "fail: 3 mismatched"}),
    ]
    s = summarize(problems)
    assert (s["problems"], s["improved"], s["kept_seed"], s["regressions"]) == (4, 1, 2, 1)
    assert s["incorrect"] == 1 and s["correctness"] == {"kernelbench": {"fail": 1}}
    assert s["geomean_over_seed"] == pytest.approx((2 * 1 * 0.8) ** (1 / 4))
    assert s["geomean_kernel_over_seed"] == pytest.approx(3)
    assert s["geomean_over_eager"] == pytest.approx((4 * 2 * 1.6 * 2) ** (1 / 4))
    assert s["faster_than_compile"] == 1 and s["above_roofline"] == ["a"]
    assert s["llm"]["cache_hit_share"] == pytest.approx(0.75)
    assert s["llm"]["cost_per_problem"] == pytest.approx(1.0)  # "d" at 1.05x is not flagged


def test_roofline_us():
    device = DeviceInfo.model_construct(name="g")
    roof = Roofline(
        dram_copy_gbps=800, dram_read_gbps=1000, l2_copy_gbps=3000, tflops={"float16": 100}
    )
    profile = DeviceProfile.model_construct(device=device, roofline=roof)
    assert roofline_us(2e12, 1e9, "float16", profile) == pytest.approx(20000)
    assert roofline_us(None, 1e9, "float16", profile) == pytest.approx(1000)
    assert roofline_us(2e12, None, "int8", profile) is None


def test_report(tmp_path):
    for name, config, final in (
        ("base", CONFIG, 50),
        ("nokb", CONFIG.model_copy(update={"kb": False, "max_iters": 5}), 80),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "config.json").write_text(config.model_dump_json())
        (run_dir / "device.json").write_text('{"gpu": "g"}')
        save(problem_eval("a", 100, final), run_dir)
    assert load_run(tmp_path / "base")[1][0].final_us == 50
    text = report({"base": tmp_path / "base", "nokb": tmp_path / "nokb"})
    assert "| base | baseline | 1 | 1 | 0 | 0 | 2.00x | 4.00x |" in text
    assert "| nokb | no KB, T = 5 | 1 | 1 | 0 | 0 | 1.25x |" in text
    assert "### nokb" in text and "| a | 100.0 us | 80.0 us | 1.25x |" in text
    assert differences(CONFIG.model_copy(update={"best_of": 3}), CONFIG) == "best-of-3"


@pytest.mark.gpu
def test_evaluate(monkeypatch, tmp_path):
    problem = copy_problem(tmp_path)
    good = KERNEL.read_text()
    problem.seed.write_text(slow(good))
    llm = LLMClient(LLMConfig())

    def send(request):
        schema = request["output_config"].get("format", {}).get("schema", {})
        if "summary" in schema.get("properties", {}):
            return message(json.dumps({"summary": "tiles are too small", "issues": []}))
        if "stages" in schema.get("properties", {}):
            return message(json.dumps({"stages": ["autotuning"], "reason": "tiles"}))
        return message(f"```python\n{good}```")

    monkeypatch.setattr(llm, "_send", send)
    config = Config(cache_dir=tmp_path / "cache", timing=TimingConfig(warmup=3, iters=40, rounds=4))
    run = CONFIG.model_copy(update={"best_of": 2, "ncu": False, "memcheck": False})
    with Runner(config) as runner:
        result = evaluate(problem, runner, llm, load_kb(), None, run, tmp_path / "w")
    assert not result.kept_seed and result.speedup > 1.5 and len(result.runs) == 2
    assert result.accepted == ["autotuning"] and result.llm.calls == 4  # analyzer, optimizer x2
    assert result.correctness == {"ci": "pass", "bench-gpu": "pass"}
    assert result.kernel_speedup > 1.5 and result.wall_s > 0
    assert (tmp_path / "w" / "run-1" / "pipeline.json").exists()
