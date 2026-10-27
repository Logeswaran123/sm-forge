"""Runtime configuration from environment variables and ``.env``."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class TimingConfig(BaseModel):
    warmup: int = 100
    iters: int = 200
    rounds: int = 10
    flush_mb: int = 256
    quantile: float = 0.2
    min_effect: float = 0.02


class CorrectnessConfig(BaseModel):
    rtol: float | None = None
    atol: float | None = None
    draws: int = 2


class Config(BaseModel):
    device: int = 0
    worker_timeout: float = 600.0
    cache_dir: Path = Path(".smforge")
    hardware_dir: Path = Path("hardware")
    timing: TimingConfig = Field(default_factory=TimingConfig)
    correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()
        return cls(
            **_read(
                device=("DEVICE_INDEX", int),
                worker_timeout=("WORKER_TIMEOUT", float),
                cache_dir=("CACHE_DIR", Path),
                hardware_dir=("HARDWARE_DIR", Path),
            ),
            timing=TimingConfig(
                **_read(
                    warmup=("BENCH_WARMUP", int),
                    iters=("BENCH_ITERS", int),
                    rounds=("BENCH_ROUNDS", int),
                    flush_mb=("BENCH_FLUSH_MB", int),
                    quantile=("BENCH_QUANTILE", float),
                    min_effect=("BENCHMARK_MIN_EFFECT", float),
                )
            ),
            correctness=CorrectnessConfig(
                **_read(
                    rtol=("CORRECTNESS_RTOL", float),
                    atol=("CORRECTNESS_ATOL", float),
                    draws=("CORRECTNESS_DRAWS", int),
                )
            ),
        )


def _read(**fields: tuple[str, type]) -> dict:
    values = {}
    for field, (name, cast) in fields.items():
        raw = os.environ.get(name, "").strip()
        if raw:
            values[field] = cast(raw)
    return values


@cache
def get_config() -> Config:
    return Config.from_env()
