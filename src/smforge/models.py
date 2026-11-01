"""Data models shared by the harness, the device profile and the CLIs."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Stage(StrEnum):
    """Optimization stages, in the order the planner considers them."""

    ALGORITHMIC = "algorithmic"
    DTYPE_FIX = "dtype_fix"
    FUSION = "fusion"
    MEMORY_ACCESS = "memory_access"
    TENSOR_DESCRIPTORS = "tensor_descriptors"
    PERSISTENT_KERNEL = "persistent_kernel"
    DEVICE_SPECIFIC = "device_specific"
    OPEN_ENDED = "open_ended"
    AUTOTUNING = "autotuning"


class IssueType(StrEnum):
    """What can be wrong with a kernel. The first group is detected from facts, ncu counters or
    the source; the second is judged by the analyzer."""

    REGISTER_SPILL = "register_spill"
    NO_TENSOR_CORE_MMA = "no_tensor_core_mma"
    TF32_PRECISION_RISK = "tf32_precision_risk"
    LOW_OCCUPANCY = "low_occupancy"
    SMEM_OVER_BUDGET = "smem_over_budget"
    MISSING_PIPELINING = "missing_pipelining"
    GRID_UNDERFILL = "grid_underfill"
    TAIL_WAVE = "tail_wave"
    LAUNCH_BOUND = "launch_bound"
    UNCOALESCED_ACCESS = "uncoalesced_access"
    BANK_CONFLICT = "bank_conflict"
    HOST_SYNC_IN_FORWARD = "host_sync_in_forward"
    UNTUNED_CONFIG = "untuned_config"
    ALGEBRAIC_SIMPLIFICATION = "algebraic_simplification"
    UNFUSED_OPS = "unfused_ops"
    UPCAST_OPERANDS = "upcast_operands"
    PER_CALL_WEIGHT_PREP = "per_call_weight_prep"
    SPLIT_K_OPPORTUNITY = "split_k_opportunity"
    OPEN_ENDED = "open_ended"


class Issue(BaseModel):
    type: IssueType
    evidence: str
    kernel: str | None = None
    source: str = "llm"


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
    oracle_dtype: str | None = None
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
    """Kernels per forward. ``span_us`` runs from the first kernel's start to the last one's end
    in a forward of its own; ``idle_us`` is the part of it no kernel was running."""

    launches: list[Launch]
    kernel_us: float
    span_us: float | None = None
    idle_us: float | None = None


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


class NcuKernel(BaseModel):
    """Nsight Compute counters for one kernel, averaged over its launches in one forward."""

    name: str
    launches: int
    duration_us: float
    sm_pct: float
    dram_pct: float
    load_bytes_per_sector: float | None = None
    store_bytes_per_sector: float | None = None
    shared_conflicts_per_wavefront: float | None = None


class NcuMetric(BaseModel):
    section: str
    name: str
    unit: str
    value: str


class NcuRule(BaseModel):
    """An ncu rule result. ``kind`` is OPT for an optimization hint; ``speedup`` is ncu's
    estimate in percent, for the whole kernel when ``scope`` is global."""

    section: str
    name: str
    kind: str
    scope: str = ""
    speedup: float | None = None
    description: str = ""


class NcuProfile(BaseModel):
    """ncu sections for one launch of one kernel."""

    kernel: str
    grid: str = ""
    block: str = ""
    metrics: list[NcuMetric] = Field(default_factory=list)
    rules: list[NcuRule] = Field(default_factory=list)

    def values(self) -> dict[tuple[str, str], tuple[str, str]]:
        return {(m.section, m.name): (m.value, m.unit) for m in self.metrics}


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


class ReferenceProfile(BaseModel):
    """One forward of a reference model: footprint, time, work, and whether its result holds up."""

    peak_bytes: int
    time_us: float
    flop: float | None = None
    bytes_moved: float
    finite: bool
    stable: bool | None = None
    max_abs_error: float | None = None


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


class SanitizerReport(BaseModel):
    """One compute-sanitizer run. ``detail`` says why the program failed without an error."""

    tool: str
    ok: bool
    errors: int
    messages: list[str] = Field(default_factory=list)
    detail: str | None = None


class DeviceProfile(BaseModel):
    device: DeviceInfo
    probes: list[ProbeResult] = Field(default_factory=list)
    roofline: Roofline | None = None
    noise: NoiseResult | None = None
    tools: list[ToolStatus] = Field(default_factory=list)

    @property
    def features(self) -> dict[str, bool]:
        return {probe.name: probe.supported for probe in self.probes}
