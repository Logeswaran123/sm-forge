"""Data models shared by the harness, the device profile and the CLIs."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    check: str
    severity: Severity
    message: str
    line: int | None = None


class OutputDiff(BaseModel):
    index: int
    shape: list[int]
    total: int
    mismatched: int
    max_abs: float
    max_rel: float
    nan: int = 0
    inf: int = 0
    first_bad: list[int] | None = None


class CheckResult(BaseModel):
    passed: bool
    rtol: float
    atol: float
    draws: int = 0
    weights: str | None = None
    diffs: list[OutputDiff] = Field(default_factory=list)
    error: str | None = None


class LaunchCategory(StrEnum):
    TRITON = "triton"
    VENDOR = "vendor"
    ATEN_COMPUTE = "aten_compute"
    ATEN_DATA = "aten_data"
    MEMORY = "memory"
    OTHER = "other"


class Launch(BaseModel):
    name: str
    category: LaunchCategory
    calls: float
    duration_us: float


class LaunchProfile(BaseModel):
    launches: list[Launch]
    kernel_us: float


class Timing(BaseModel):
    time_us: float
    quantile: float
    median_us: float
    mean_us: float
    min_us: float
    max_us: float
    samples: int
    graph: bool = False
    tflops: float | None = None
    gbps: float | None = None


class Verdict(StrEnum):
    FASTER = "faster"
    SLOWER = "slower"
    PARITY = "parity"


class Comparison(BaseModel):
    baseline_us: float
    candidate_us: float
    speedup: float
    verdict: Verdict
    min_effect: float
    quantile: float
    ratios: list[float]
    graph: bool = False


class KernelFacts(BaseModel):
    name: str
    grid: list[int]
    ctas: int
    num_warps: int
    num_stages: int
    n_regs: int
    n_spills: int
    shared_bytes: int
    ctas_per_sm: int | None = None
    waves: float | None = None
    mma: dict[str, int] = Field(default_factory=dict)
    instructions: dict[str, int] = Field(default_factory=dict)
    total_instructions: int = 0


class FactsReport(BaseModel):
    kernels: list[KernelFacts]
    flop: float | None = None
    bytes_moved: float | None = None

    @property
    def intensity(self) -> float | None:
        if self.flop and self.bytes_moved:
            return self.flop / self.bytes_moved
        return None


class DeviceInfo(BaseModel):
    index: int
    name: str
    capability: tuple[int, int]
    sm_count: int
    memory_bytes: int
    l2_bytes: int
    smem_per_block: int
    smem_per_sm: int
    regs_per_sm: int
    max_threads_per_sm: int
    max_ctas_per_sm: int | None = None
    warp_size: int
    memory_bus_bits: int
    max_sm_clock_mhz: int | None = None
    max_memory_clock_mhz: int | None = None
    power_limit_w: float | None = None
    driver: str | None = None
    cuda: str | None = None
    torch: str
    triton: str
    wsl: bool = False

    @property
    def arch(self) -> str:
        return f"sm_{self.capability[0]}{self.capability[1]}"

    @property
    def slug(self) -> str:
        name = self.name.lower().replace("nvidia", "").replace("geforce", "")
        return re.sub(r"[^a-z0-9]+", "", name)


class ProbeResult(BaseModel):
    name: str
    compiled: bool
    lowered: bool | None = None
    correct: bool | None = None
    instructions: dict[str, int] = Field(default_factory=dict)
    error: str | None = None

    @property
    def supported(self) -> bool:
        return self.compiled and self.lowered is not False and self.correct is not False


class Roofline(BaseModel):
    dram_copy_gbps: float
    dram_read_gbps: float
    l2_copy_gbps: float
    tflops: dict[str, float]

    def ridge(self, dtype: str) -> float | None:
        """Arithmetic intensity (FLOP/byte) where the dtype's peak meets DRAM read bandwidth."""
        peak = self.tflops.get(dtype)
        return peak * 1e3 / self.dram_read_gbps if peak else None


class NoiseCase(BaseModel):
    name: str
    time_us: float
    deviation: float
    spread: float


class NoiseResult(BaseModel):
    cases: list[NoiseCase]
    min_effect: float


class ToolStatus(BaseModel):
    name: str
    path: str | None = None
    ok: bool
    detail: str = ""


class DeviceProfile(BaseModel):
    device: DeviceInfo
    probes: list[ProbeResult] = Field(default_factory=list)
    roofline: Roofline | None = None
    noise: NoiseResult | None = None
    tools: list[ToolStatus] = Field(default_factory=list)

    @property
    def features(self) -> dict[str, bool]:
        return {probe.name: probe.supported for probe in self.probes}
