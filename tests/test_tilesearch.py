import pytest

from smforge.harness.audit import audit_source
from smforge.harness.static import validate_source
from smforge.models import Severity
from smforge.spec import load_spec
from smforge.tilesearch import (
    GemmConfig,
    GemmShape,
    alignment,
    default_config,
    evaluate,
    render,
    validate,
    write_problem,
)

SMEM = 101376


def config(tile=(128, 128, 32), warp=(64, 64, 32), stages=4, **kw) -> GemmConfig:
    return GemmConfig(tile=tile, warp=warp, stages=stages, **kw)


def test_shape_and_alignment():
    shape = GemmShape.parse("8205x5921x2949:bfloat16:row")
    assert (shape.m, shape.n, shape.k, shape.dtype, shape.b_layout) == (
        8205,
        5921,
        2949,
        "bfloat16",
        "row",
    )
    assert [alignment(x) for x in (4096, 2949, 5922, 100)] == [8, 1, 2, 4]


def test_validate():
    shape = GemmShape(m=4096, n=4096, k=4096)
    assert validate(config(), shape, SMEM) is None
    assert "divide" in validate(config(warp=(48, 64, 32)), shape, SMEM)
    assert "instruction" in validate(config(tile=(120, 128, 32), warp=(40, 64, 32)), shape, SMEM)
    assert "accumulators" in validate(config(warp=(128, 64, 32)), shape, SMEM)
    assert "144 KB" in validate(config((128, 256, 64), (64, 64, 64), 3), shape, SMEM)
    assert "split_k" in validate(config(split_k=256), shape, SMEM)
    default, fitted = default_config(SMEM)
    assert default.stages == 3 and fitted.stages == 2 and validate(fitted, shape, SMEM) is None


def test_rendered_file_passes_static_checks():
    for shape in (GemmShape(m=256, n=256, k=4096), GemmShape(m=100, n=6, k=10, b_layout="row")):
        code = render(config(split_k=2, swizzle=4), shape)
        issues = validate_source(code) + audit_source(code)
        assert [i for i in issues if i.severity is Severity.ERROR] == []
    assert "GemmSplitKParallel" in code and "4, 2, 2>;" in code
    plain = render(config(swizzle=4), GemmShape(m=256, n=256, k=4096))
    assert "GemmIdentityThreadblockSwizzle<4>" in plain and "4, 8, 8>;" in plain


def test_problem_spec(tmp_path):
    root = write_problem(GemmShape(m=8205, n=5921, k=2949), tmp_path)
    spec = load_spec(root / "spec.yaml")
    ci = spec.variant("ci")
    assert ci.dims == {"M": 127, "N": 193, "K": 101}
    assert spec.variant("bench-gpu").dims == {"M": 8205, "N": 5921, "K": 2949}


@pytest.mark.gpu
def test_evaluate(tmp_path):
    from smforge.config import Config, TimingConfig
    from smforge.harness.runner import Runner

    shape = GemmShape(m=512, n=384, k=1024)
    root = write_problem(shape, tmp_path)
    timing = TimingConfig(warmup=3, iters=20, rounds=2)
    with Runner(Config(timing=timing)) as runner:
        trial = evaluate(config(split_k=2), shape, root, runner, SMEM)
        assert trial.status == "ok" and trial.us > 0, trial.detail
        row = evaluate(
            config((64, 64, 32), (32, 32, 32), 3),
            GemmShape(m=512, n=384, k=1024, b_layout="row"),
            write_problem(GemmShape(m=512, n=384, k=1024, b_layout="row"), tmp_path / "row"),
            runner,
            SMEM,
        )
        assert row.status == "ok", row.detail
        assert evaluate(config(warp=(128, 64, 32)), shape, root, runner, SMEM).status == "invalid"
