import pytest
import yaml

from smforge.spec import evaluate, parse_spec

XE_FORGE_SPEC = """
inputs:
  X:
    shape: [BATCH, INPUT_SIZE]
    dtype: float16

inits:
  - dim: INPUT_SIZE
  - dim: HIDDEN_SIZE
  - dim: OUTPUT_SIZE

ci:
  - params: [X]
    dtype: float16
    dims:
      BATCH: 16384
      INPUT_SIZE: 2048
      HIDDEN_SIZE: 4096
      OUTPUT_SIZE: 1024
    flop: "2*BATCH*HIDDEN_SIZE*INPUT_SIZE + 2*BATCH*OUTPUT_SIZE*HIDDEN_SIZE"

bench-gpu:
  - params: [X]
    dtype: float16
    dims:
      BATCH: 8
      INPUT_SIZE: 16
      HIDDEN_SIZE: 32
      OUTPUT_SIZE: 4
    flop: "2*BATCH*HIDDEN_SIZE*INPUT_SIZE + 2*BATCH*OUTPUT_SIZE*HIDDEN_SIZE"
    rtol: 1e-2
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


def test_xe_forge_spec():
    spec = parse_spec(yaml.safe_load(XE_FORGE_SPEC))
    assert sorted(spec.variants) == ["bench-gpu", "ci"]
    variant = spec.variant()
    assert spec.input_shapes(variant) == [(8, 16)]
    assert spec.init_args(variant) == [16, 32, 4]
    assert spec.flop(variant) == 2 * 8 * 32 * 16 + 2 * 8 * 4 * 32
    assert variant.rtol == pytest.approx(1e-2)
    assert spec.dtype(variant) == "float16"


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
