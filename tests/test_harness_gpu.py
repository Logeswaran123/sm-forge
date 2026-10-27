from pathlib import Path

import pytest

from smforge.config import Config, TimingConfig
from smforge.harness.audit import audit_launches, audit_source
from smforge.harness.runner import Runner, WorkerError
from smforge.models import LaunchCategory

pytestmark = pytest.mark.gpu

PROBLEM = Path(__file__).parent / "kernels" / "linear_relu"
SPEC = PROBLEM / "spec.yaml"
REFERENCE = PROBLEM / "reference.py"
KERNEL = PROBLEM / "kernel.py"


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    config = Config(
        cache_dir=tmp_path_factory.mktemp("cache"),
        timing=TimingConfig(warmup=10, iters=40, rounds=4, flush_mb=128),
    )
    with Runner(config) as runner:
        yield runner


def edited(tmp_path: Path, old: str, new: str) -> Path:
    code = KERNEL.read_text()
    assert old in code
    path = tmp_path / "candidate.py"
    path.write_text(code.replace(old, new))
    return path


@pytest.mark.parametrize("variant", ["ci", "bench-gpu"])
def test_kernel_matches_reference(runner, variant):
    result = runner.check(KERNEL, REFERENCE, spec=SPEC, variant=variant)
    assert result.passed, result
    assert result.draws == 2
    assert result.weights == "by_position"
    assert (result.rtol, result.atol) == (1e-2, 1e-2)


def test_wrong_mask(runner, tmp_path):
    candidate = edited(tmp_path, "(offs_n[None, :] < N)\n", "(offs_n[None, :] < N - 1)\n")
    result = runner.check(candidate, REFERENCE, spec=SPEC, variant="ci")
    assert not result.passed
    diff = result.diffs[0]
    assert diff.mismatched > 0
    assert diff.first_bad[1] == 199


def test_fault_restarts_worker(runner, tmp_path):
    candidate = edited(tmp_path, "y = y_ptr + offs_m", "y = y_ptr + (1 << 40) + offs_m")
    result = runner.check(candidate, REFERENCE, spec=SPEC, variant="ci")
    assert not result.passed
    assert "illegal memory access" in result.error
    assert runner.check(KERNEL, REFERENCE, spec=SPEC, variant="ci").passed


def test_compile_error_is_not_fatal(runner, tmp_path):
    pid = runner.call("ping")["pid"]
    candidate = edited(tmp_path, "tl.dot(x, w, acc)", "tl.dott(x, w, acc)")
    result = runner.check(candidate, REFERENCE, spec=SPEC, variant="ci")
    assert not result.passed
    assert "dott" in result.error
    assert runner.call("ping")["pid"] == pid


def test_cached_output_fails_second_draw(runner, tmp_path):
    candidate = edited(
        tmp_path,
        "    def forward(self, x):\n",
        "    def forward(self, x):\n"
        "        if not hasattr(self, 'out'):\n"
        "            self.out = self.compute(x)\n"
        "        return self.out\n\n"
        "    def compute(self, x):\n",
    )
    result = runner.check(candidate, REFERENCE, spec=SPEC, variant="ci")
    assert not result.passed
    assert result.draws == 2


def test_pytorch_fallback_is_flagged(runner, tmp_path):
    candidate = edited(
        tmp_path,
        "        return y\n",
        "        return torch.relu(torch.nn.functional.linear(x, self.weight, self.bias))\n",
    )
    assert runner.check(candidate, REFERENCE, spec=SPEC, variant="ci").passed
    assert {i.check for i in audit_source(candidate.read_text())} == {"torch_compute_call"}
    baseline = runner.launches(KERNEL, spec=SPEC, variant="ci")
    launches = runner.launches(candidate, spec=SPEC, variant="ci")
    assert {x.category for x in launches.launches} >= {
        LaunchCategory.TRITON,
        LaunchCategory.VENDOR,
    }
    checks = {i.check for i in audit_launches(launches, baseline)}
    assert "vendor_launch" in checks
    assert audit_launches(baseline) == []


def test_bench_and_compare(runner):
    timing = runner.bench(KERNEL, spec=SPEC)
    assert timing.time_us > 0
    assert timing.samples == 40
    assert timing.tflops and timing.gbps
    comparison = runner.compare(KERNEL, KERNEL, spec=SPEC)
    assert 0.9 < comparison.speedup < 1.1
    assert len(comparison.ratios) == 4


def test_graph_mode(runner):
    timing = runner.bench(KERNEL, spec=SPEC, graph=True)
    assert timing.graph
    assert timing.time_us > 0


def test_facts(runner):
    report = runner.facts(KERNEL, spec=SPEC)
    (kernel,) = report.kernels
    assert kernel.name == "linear_relu_kernel"
    assert kernel.grid == [64, 32, 1]
    assert kernel.ctas == 2048
    assert kernel.n_regs > 0
    assert kernel.n_spills == 0
    assert sum(kernel.mma.values()) > 0
    assert kernel.ctas_per_sm
    assert report.intensity == pytest.approx(4096 * 2048 / (4096 + 2048 + 4096))


def test_timeout_restarts_worker(runner):
    pid = runner.call("ping")["pid"]
    with pytest.raises(WorkerError, match="timed out") as error:
        runner.call(
            "bench",
            timeout=0.01,
            kernel=str(KERNEL),
            timing=runner.config.timing.model_dump(),
            spec=str(SPEC),
            variant="bench-gpu",
        )
    assert error.value.fatal
    assert runner.call("ping")["pid"] != pid
