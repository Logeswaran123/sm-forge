import pytest
from test_seed import PROBLEM, copy_problem

from smforge.agents.candidate import check_candidate, float32_note, make_base
from smforge.config import Config, TimingConfig
from smforge.harness.runner import Runner
from smforge.spec import load_spec

pytestmark = pytest.mark.gpu


def test_float32_regression_is_rejected(tmp_path):
    problem = copy_problem(tmp_path)
    spec_text = problem.spec.read_text().replace("[M, K], dtype: float16", "[M, K], dtype: float32")
    problem.spec.write_text(spec_text)
    tf32 = (PROBLEM / "kernel.py").read_text()
    ieee = tf32.replace("tl.dot(x, w, acc)", 'tl.dot(x, w, acc, input_precision="ieee")')
    problem.seed.write_text(ieee)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(tf32)
    config = Config(cache_dir=tmp_path / "cache", timing=TimingConfig(warmup=3, iters=20, rounds=2))
    with Runner(config) as runner:
        base = make_base(runner, problem, problem.seed)
        assert base.float32 and "float32" in float32_note(base)
        outcome = check_candidate(runner, problem, load_spec(problem.spec), base, candidate)
        assert outcome.kind == "float32", outcome.feedback
        assert 'input_precision="ieee"' in outcome.feedback
        reverse = make_base(runner, problem, candidate)
        assert not reverse.float32 and float32_note(reverse) is None
        outcome = check_candidate(runner, problem, load_spec(problem.spec), reverse, problem.seed)
        assert outcome.correct


def test_mismatch_gets_sanitizer_findings(tmp_path):
    problem = copy_problem(tmp_path)
    good = (PROBLEM / "kernel.py").read_text()
    problem.seed.write_text(good)
    old = "y = y_ptr + offs_m[:, None] * stride_ym"
    assert old in good
    shifted = tmp_path / "shifted.py"
    shifted.write_text(good.replace(old, "y = y_ptr + (offs_m[:, None] + 1) * stride_ym"))
    config = Config(cache_dir=tmp_path / "cache", timing=TimingConfig(warmup=3, iters=20, rounds=2))
    spec = load_spec(problem.spec)
    with Runner(config) as runner:
        base = make_base(runner, problem, problem.seed)
        outcome = check_candidate(runner, problem, spec, base, shifted, sanitize_failures=True)
    assert outcome.kind == "mismatch"
    if "compute-sanitizer on `ci`" not in outcome.feedback:
        pytest.skip("compute-sanitizer does not run here")
    assert "memcheck:" in outcome.feedback and "Invalid __global__ write" in outcome.feedback
