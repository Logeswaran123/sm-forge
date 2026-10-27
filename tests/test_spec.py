import pytest
import yaml

from smforge.spec import evaluate, parse_spec

CONV_SPEC = """
inputs:
  X:
    shape: [N, C_IN, H, W]
    dtype: bfloat16

inits:
  - in_channels: C_IN
  - out_channels: C_OUT
  - kernel_size: KS

ci:
  - params: [X]
    dtype: bfloat16
    dims: {N: 3, C_IN: 5, C_OUT: 7, H: 9, W: 11, KS: 3}

bench-gpu:
  - params: [X]
    dtype: bfloat16
    dims: {N: 4, C_IN: 8, C_OUT: 16, H: 10, W: 12, KS: 3}
    flop: "2*N*C_OUT*(H - KS + 1)*(W - KS + 1)*C_IN*KS*KS"
    atol: 0.05
"""

MOE_SPEC = """
inputs:
  A: {shape: [M, K], dtype: bfloat16}
  topk_ids: {shape: [M, TOPK], dtype: int32, range: [0, E]}
inits:
  - experts: E
  - quant: none
  - bias_shape: [E, 1, 1]
  - scale: 0.5
default_variant: decode
decode:
  - params: [A, topk_ids]
    dtype: float16
    dims: {M: 4, K: 64, TOPK: 2, E: 8}
    bytes: "2*M*K + 4*M*TOPK"
"""


def test_default_variant_is_bench_gpu():
    spec = parse_spec(yaml.safe_load(CONV_SPEC))
    assert sorted(spec.variants) == ["bench-gpu", "ci"]
    variant = spec.variant()
    assert spec.input_shapes(variant) == [(4, 8, 10, 12)]
    assert spec.init_args(variant) == [8, 16, 3]
    assert spec.flop(variant) == 2 * 4 * 16 * 8 * 10 * 8 * 3 * 3
    assert variant.atol == pytest.approx(0.05)
    assert spec.dtype(variant) == "bfloat16"


def test_dtype_override_skips_integer_inputs():
    spec = parse_spec(yaml.safe_load(MOE_SPEC))
    variant = spec.variant()
    assert spec.resolve() == "decode"
    assert spec.input_dtypes(variant) == ["float16", "int32"]
    assert spec.input_ranges(variant) == [None, (0, 8)]
    assert spec.init_args(variant) == [8, "none", (8, 1, 1), 0.5]
    assert spec.bytes_moved(variant) == 2 * 4 * 64 + 4 * 4 * 2
    assert spec.flop(variant) is None


def test_unknown_variant():
    spec = parse_spec(yaml.safe_load(MOE_SPEC))
    with pytest.raises(KeyError, match="decode"):
        spec.resolve("bench-gpu")


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("2*M*N", 16.0),
        ("M**2 - N // 2", 2.0),
        ("max(M, N) / 2", 2.0),
        ("-M + 1", -1.0),
        (7, 7.0),
    ],
)
def test_evaluate(expr, value):
    assert evaluate(expr, {"M": 2, "N": 4}) == value


@pytest.mark.parametrize("expr", ["__import__('os')", "M.real", "K * 2"])
def test_evaluate_rejects(expr):
    with pytest.raises(ValueError):
        evaluate(expr, {"M": 2})
