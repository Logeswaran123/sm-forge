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


class LLMConfig(BaseModel):
    backend: str = "api"
    claude_bin: str | None = None
    model: str = "claude-opus-5-5"
    effort: str = "high"
    efforts: dict[str, str] = Field(default_factory=lambda: {"planner": "low"})
    max_tokens: int = 64000
    max_retries: int = 4
    budget_usd: float | None = None

    def effort_for(self, role: str | None) -> str:
        """The effort for a role (seed, optimizer, analyzer, planner), else the default."""
        return self.efforts.get(role, self.effort) if role else self.effort


class Config(BaseModel):
    device: int = 0
    worker_timeout: float = 600.0
    cache_dir: Path = Path(".smforge")
    hardware_dir: Path = Path("hardware")
    timing: TimingConfig = Field(default_factory=TimingConfig)
    correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

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
            llm=LLMConfig(
                **_read(
                    backend=("LLM_BACKEND", str),
                    claude_bin=("CLAUDE_BIN", str),
                    model=("LLM_MODEL", str),
                    effort=("LLM_EFFORT", str),
                    max_tokens=("LLM_MAX_TOKENS", int),
                    budget_usd=("LLM_BUDGET_USD", float),
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
