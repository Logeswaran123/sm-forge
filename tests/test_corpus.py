from pathlib import Path

import pytest
import yaml

from smforge.config import Config, TimingConfig
from smforge.corpus.build import import_kernelbench
from smforge.corpus.catalog import Problem, load_problems, make_splits
from smforge.corpus.kernelbench import Untranslatable, family, problem_id, translate
from smforge.corpus.results import load_results, upsert
from smforge.harness.runner import Runner
from smforge.spec import load_spec

PROBLEM = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = scale

    def forward(self, x, y):
        return torch.relu(self.linear(x) + self.bias) * self.scale + y


batch_size = 64
in_features = 128
out_features = 2 * 64
bias_shape = (out_features,)
scale = 0.5


def get_inputs():
    x = torch.rand(batch_size, in_features)
    return [x, torch.randn((batch_size, out_features))]


def get_init_inputs():
    return [in_features, out_features, bias_shape, scale]
"""


@pytest.fixture
def kernelbench_file(tmp_path):
    path = tmp_path / "7_Gemm_ReLU_Scale.py"
    path.write_text(PROBLEM)
    return path


def test_translate(kernelbench_file):
    sections, dims = translate(kernelbench_file)
    assert dims == {"BATCH_SIZE": 64, "IN_FEATURES": 128, "OUT_FEATURES": 128}
    assert sections["inputs"] == {
        "x": {"shape": ["BATCH_SIZE", "IN_FEATURES"], "dtype": "float32", "range": [0.0, 1.0]},
        "x1": {"shape": ["BATCH_SIZE", "OUT_FEATURES"], "dtype": "float32"},
    }
    assert sections["inits"] == [
        {"in_features": "IN_FEATURES"},
        {"out_features": "OUT_FEATURES"},
        {"bias_shape": ["OUT_FEATURES"]},
        {"scale": 0.5},
    ]
    assert problem_id(2, kernelbench_file) == "l2_007_gemm_relu_scale"
    assert family(2, kernelbench_file) == "gemm"


def test_untranslatable_input(tmp_path):
    path = tmp_path / "1_Tri.py"
    path.write_text(PROBLEM.replace("torch.rand(batch_size, in_features)", "torch.triu(t)"))
    with pytest.raises(Untranslatable, match="triu"):
        translate(path)


def problems(*families):
    return [
        Problem(id=f"{f}_{i}", source="test", family=f, path=Path("."))
        for i, f in enumerate(families)
    ]


def test_splits_are_stratified_and_frozen():
    corpus = problems(*["gemm"] * 10, *["conv2d"] * 10)
    splits = make_splits(corpus, {}, eval_fraction=0.3)
    assert len(splits["eval"]) == 6
    assert sum(pid.startswith("gemm") for pid in splits["eval"]) == 3
    grown = make_splits([*corpus, *problems("bmm", "bmm", "bmm")], splits, eval_fraction=0.3)
    assert set(splits["eval"]) <= set(grown["eval"])
    assert set(splits["dev"]) <= set(grown["dev"])


def test_results_upsert(tmp_path):
    path = tmp_path / "baselines.csv"
    row = {"problem": "p", "variant": "bench-gpu", "mode": "eager", "time_us": "10"}
    upsert([row, {**row, "mode": "compile", "time_us": "8"}], path)
    upsert([{**row, "time_us": "9"}], path)
    times = {r["mode"]: r["time_us"] for r in load_results(path)}
    assert times == {"compile": "8", "eager": "9"}


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    config = Config(
        cache_dir=tmp_path_factory.mktemp("cache"),
        timing=TimingConfig(warmup=3, iters=10, rounds=2, flush_mb=64),
    )
    with Runner(config) as runner:
        yield runner


@pytest.mark.gpu
def test_import_kernelbench(runner, kernelbench_file, tmp_path):
    problem = import_kernelbench(kernelbench_file, 2, tmp_path / "corpus", runner, "test@0")
    (loaded,) = load_problems(tmp_path / "corpus")
    assert loaded.id == problem.id == "l2_007_gemm_relu_scale"
    assert loaded.reference.read_text() == PROBLEM
    spec = load_spec(problem.spec)
    assert list(spec.variants) == ["bench-gpu", "ci", "kernelbench"]
    bench = spec.variant()
    assert spec.dtype(bench) == "float16"
    assert bench.flop == 2 * 64 * 128 * 128
    assert spec.variant("ci").dims["BATCH_SIZE"] == 3
    assert "dtype" not in yaml.safe_load(problem.spec.read_text())["kernelbench"][0]
    profile = runner.profile_reference(problem.reference, spec=problem.spec)
    assert profile.finite and profile.stable
    for mode in ("eager", "compile"):
        assert runner.baseline(problem.reference, mode, spec=problem.spec).time_us > 0


def test_scale_dim():
    from smforge.corpus.build import scale_dim

    matmul = {"inputs": {"A": {"shape": ["M", "K"]}, "B": {"shape": ["K", "N"]}}, "inits": []}
    assert scale_dim(matmul, {"M": 8, "K": 4, "N": 2}) == "M"
    batched = {**matmul, "inits": [{"bias_shape": ["M", 1]}]}
    assert scale_dim(batched, {"BATCH_SIZE": 4, "M": 8}) == "BATCH_SIZE"
    assert scale_dim(batched, {"M": 8, "K": 4}) is None
