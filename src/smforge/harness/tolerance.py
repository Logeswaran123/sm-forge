"""Correctness tolerances: CLI, then spec variant, then environment, then a dtype default."""

from __future__ import annotations

from smforge.config import CorrectnessConfig
from smforge.spec import Variant

DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "float64": (1e-7, 1e-7),
    "float32": (1e-4, 1e-4),
    "float16": (1e-2, 1e-2),
    "bfloat16": (2e-2, 2e-2),
    "float8_e4m3fn": (1e-1, 1e-1),
    "float8_e5m2": (2e-1, 2e-1),
}
INTEGER_TOLERANCE = (0.0, 0.0)


def resolve_tolerance(
    dtype: str,
    variant: Variant | None = None,
    config: CorrectnessConfig | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[float, float]:
    default = DEFAULT_TOLERANCES.get(dtype, INTEGER_TOLERANCE)
    rtols = (rtol, variant and variant.rtol, config and config.rtol, default[0])
    atols = (atol, variant and variant.atol, config and config.atol, default[1])
    return (
        next(v for v in rtols if v is not None),
        next(v for v in atols if v is not None),
    )
