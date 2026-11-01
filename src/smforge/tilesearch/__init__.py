"""CUTLASS tile search: the LLM proposes GEMM configs, the harness checks and times them.

A config is CUTLASS's SM80 device GEMM (``mma.sync`` 16x8x16, ``cp.async`` multistage) with a
threadblock tile, a warp tile, a stage count, a raster swizzle and split-K. Split-K uses
CUTLASS's parallel form, float partials summed by a second kernel: the serial form adds each
slice into the fp16 output, which misses the tolerance where large partials cancel. On sm_120,
CUTLASS's SM120 collectives take only fp8, fp6 and fp4 inputs, so fp16 and bf16 GEMMs run these
SM80 kernels; CUTLASS's default config (128x256x64 tile, 64x64x64 warps, 3 stages) needs 144 KB
of shared memory and does not launch where a block gets 99 KB. The baseline is therefore that
tile and warp shape with as many stages as fit.

Each shape becomes a small problem (``reference.py`` computing the matmul with PyTorch, so with
cuBLAS, and a spec whose ``ci`` variant keeps the shape's alignment at odd sizes), and each config
a ``cuda`` kernel file rendered from ``templates/gemm.py.j2``. A config is validated against the
device limits before it is built; a valid one must pass ``ci`` and ``bench-gpu`` and is timed on
``bench-gpu``. Each round the LLM sees every result so far and proposes the next configs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

from smforge.harness.runner import Runner, WorkerError
from smforge.llm.client import Completer, Usage
from smforge.llm.prompting import PROMPTS
from smforge.llm.structured import ask
from smforge.models import DeviceProfile

log = logging.getLogger(__name__)

TEMPLATES = Path(__file__).parent / "templates"
INSTRUCTION = (16, 8, 16)
MAX_ACCUMULATORS = 128
MAX_THREADS = 1024
ELEMENTS = {"float16": "cutlass::half_t", "bfloat16": "cutlass::bfloat16_t"}
CUTLASS_DEFAULT = ((128, 256, 64), (64, 64, 64), 3)


class GemmShape(BaseModel):
    """C[M, N] = A[M, K] @ B, B stored ``[N, K]`` (``col``, as nn.Linear) or ``[K, N]`` (``row``)."""

    m: int
    n: int
    k: int
    dtype: str = "float16"
    b_layout: str = "col"

    @property
    def label(self) -> str:
        return f"{self.m}x{self.n}x{self.k}_{self.dtype}_{self.b_layout}"

    @property
    def flop(self) -> float:
        return 2.0 * self.m * self.n * self.k

    @classmethod
    def parse(cls, text: str) -> GemmShape:
        """``MxNxK``, optionally ``:bfloat16`` and ``:row``."""
        dims, *rest = text.split(":")
        m, n, k = (int(x) for x in dims.lower().split("x"))
        dtype = next((r for r in rest if r in ELEMENTS), "float16")
        layout = next((r for r in rest if r in ("row", "col")), "col")
        return cls(m=m, n=n, k=k, dtype=dtype, b_layout=layout)


class GemmConfig(BaseModel):
    tile: tuple[int, int, int] = Field(description="threadblock tile M, N, K")
    warp: tuple[int, int, int] = Field(description="warp tile M, N, K")
    stages: int = Field(description="cp.async pipeline stages, 2 to 6")
    swizzle: int = Field(1, description="threadblock raster swizzle: 1, 2, 4 or 8")
    split_k: int = Field(1, description="split-K slices, 1 for none; swizzle is ignored above 1")

    @property
    def label(self) -> str:
        text = f"{'x'.join(map(str, self.tile))}_w{'x'.join(map(str, self.warp))}_s{self.stages}"
        if self.swizzle > 1:
            text += f"_sw{self.swizzle}"
        if self.split_k > 1:
            text += f"_k{self.split_k}"
        return text

    @property
    def threads(self) -> int:
        return (self.tile[0] // self.warp[0]) * (self.tile[1] // self.warp[1]) * 32

    def smem_bytes(self, element_bytes: int = 2) -> int:
        return self.stages * (self.tile[0] + self.tile[1]) * self.tile[2] * element_bytes


class Proposal(BaseModel):
    configs: list[GemmConfig]
    reason: str = Field(description="one or two sentences on why these configs")


def alignment(extent: int) -> int:
    """Elements per 16-byte access the extent allows (8 for fp16), down to 1."""
    return next(a for a in (8, 4, 2, 1) if extent % a == 0)


def unsupported(shape: GemmShape) -> str | None:
    """Why no SM80 config can run this shape: ``cp.async`` copies 4, 8 or 16 bytes, so the
    rows of A and B must start on 4-byte boundaries."""
    b_extent = shape.k if shape.b_layout == "col" else shape.n
    if min(alignment(shape.k), alignment(b_extent)) < 2:
        return (
            f"rows of A ({shape.k} elements) and B ({b_extent}) must be even for cp.async's "
            "4-byte minimum copy"
        )
    return None


def validate(config: GemmConfig, shape: GemmShape, smem_limit: int) -> str | None:
    """Why the config cannot build or run on this shape and device, or None."""
    if reason := unsupported(shape):
        return reason
    (tm, tn, tk), (wm, wn, wk) = config.tile, config.warp
    if tm % wm or tn % wn or tk != wk:
        return "the warp tile must divide the threadblock tile, with the same K"
    if wm % INSTRUCTION[0] or wn % INSTRUCTION[1] or wk % INSTRUCTION[2]:
        return (
            f"the warp tile must be a multiple of the {'x'.join(map(str, INSTRUCTION))} instruction"
        )
    if not 2 <= config.stages <= 8:
        return "stages must be 2 to 8 for the multistage mainloop"
    if config.threads > MAX_THREADS or config.threads < 32:
        return f"{config.threads} threads per block"
    if wm * wn // 32 > MAX_ACCUMULATORS:
        return f"{wm * wn // 32} float accumulators per thread; at most {MAX_ACCUMULATORS} fit"
    if config.swizzle not in (1, 2, 4, 8):
        return "swizzle must be 1, 2, 4 or 8"
    if config.split_k < 1 or config.split_k > max(1, shape.k // tk):
        return f"split_k {config.split_k} leaves a slice shorter than one K tile"
    if (smem := config.smem_bytes()) > smem_limit:
        return f"needs {smem // 1024} KB of shared memory; a block gets {smem_limit // 1024} KB"
    return None


def render(config: GemmConfig, shape: GemmShape) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined)
    b_extent = shape.k if shape.b_layout == "col" else shape.n
    return env.get_template("gemm.py.j2").render(
        label=config.label,
        element=ELEMENTS[shape.dtype],
        dtype=shape.dtype,
        b_layout=shape.b_layout,
        tile=config.tile,
        warp=config.warp,
        stages=config.stages,
        swizzle=config.swizzle,
        split_k=config.split_k,
        align_a=alignment(shape.k),
        align_b=alignment(b_extent),
        align_c=alignment(shape.n),
    )


def default_config(smem_limit: int) -> tuple[GemmConfig, GemmConfig]:
    """CUTLASS's default SM80 config, and the same tile with as many stages as fit."""
    tile, warp, stages = CUTLASS_DEFAULT
    default = GemmConfig(tile=tile, warp=warp, stages=stages)
    fitted = default
    while fitted.stages > 2 and fitted.smem_bytes() > smem_limit:
        fitted = fitted.model_copy(update={"stages": fitted.stages - 1})
    return default, fitted


def write_problem(shape: GemmShape, root: Path) -> Path:
    """``reference.py`` and ``spec.yaml`` for ``shape``; ``ci`` keeps its residues mod 8."""
    root.mkdir(parents=True, exist_ok=True)
    b = "b.T" if shape.b_layout == "col" else "b"
    (root / "reference.py").write_text(
        "import torch\nimport torch.nn as nn\n\n\nclass Model(nn.Module):\n"
        f"    def forward(self, a, b):\n        return a @ {b}\n"
    )
    b_shape = "[N, K]" if shape.b_layout == "col" else "[K, N]"
    ci = {"M": 127, "N": 192 + shape.n % 8, "K": 96 + shape.k % 8}
    bench = {"M": shape.m, "N": shape.n, "K": shape.k}
    (root / "spec.yaml").write_text(
        f"inputs:\n  A: {{shape: [M, K], dtype: {shape.dtype}}}\n"
        f"  B: {{shape: {b_shape}, dtype: {shape.dtype}}}\n"
        f"ci:\n  - params: [A, B]\n    dims: {json.dumps(ci)}\n"
        f"bench-gpu:\n  - params: [A, B]\n    dims: {json.dumps(bench)}\n"
        '    flop: "2*M*N*K"\n    bytes: "2*(M*K + N*K + M*N)"\n'
    )
    return root


@dataclass
class Trial:
    config: GemmConfig
    status: str
    us: float | None = None
    detail: str = ""

    @property
    def row(self) -> str:
        time = f"{self.us:.1f} us" if self.us is not None else "-"
        return f"{self.config.label}: {self.status}, {time}" + (
            f" ({self.detail})" if self.detail else ""
        )


@dataclass
class SearchResult:
    shape: GemmShape
    default: Trial
    fitted: Trial
    cublas_us: float
    trials: list[Trial] = field(default_factory=list)
    best: Trial | None = None
    vs_default: float | None = None
    vs_cublas: float | None = None
    usage: Usage = field(default_factory=Usage)

    def tflops(self, us: float | None) -> float | None:
        return self.shape.flop / us / 1e6 if us else None


def evaluate(config: GemmConfig, shape: GemmShape, root: Path, runner: Runner, smem: int) -> Trial:
    """Validate, render, check on ``ci`` and ``bench-gpu``, and time on ``bench-gpu``."""
    if reason := validate(config, shape, smem):
        return Trial(config, "invalid", detail=reason)
    path = root / f"{config.label}.py"
    path.write_text(render(config, shape))
    problem = {"spec": root / "spec.yaml"}
    try:
        for variant in ("ci", "bench-gpu"):
            check = runner.check(path, root / "reference.py", variant=variant, **problem)
            if not check.passed:
                detail = (check.error or "outputs differ").strip().splitlines()[-1][:200]
                return Trial(config, f"failed {variant}", detail=detail)
        timing = runner.bench(path, variant="bench-gpu", **problem)
    except WorkerError as e:
        return Trial(config, "error", detail=str(e).splitlines()[0][:200])
    return Trial(config, "ok", timing.time_us)


def search(
    shape: GemmShape,
    runner: Runner,
    llm: Completer,
    profile: DeviceProfile,
    workdir: Path,
    rounds: int = 3,
    per_round: int = 6,
) -> SearchResult:
    smem = profile.device.smem_per_block
    root = write_problem(shape, workdir / shape.label)
    reference = root / "reference.py"
    cublas = runner.bench(reference, spec=root / "spec.yaml", variant="bench-gpu").time_us
    default, fitted = default_config(smem)
    default_trial = evaluate(default, shape, root, runner, smem)
    fitted_trial = (
        default_trial if fitted == default else evaluate(fitted, shape, root, runner, smem)
    )
    result = SearchResult(shape, default_trial, fitted_trial, cublas)
    system = (PROMPTS / "tilesearch.md").read_text()
    tried = {default.label, fitted.label}
    for round_index in range(rounds):
        task = _task(shape, profile, result, per_round, round_index, rounds)
        try:
            proposal, reply = ask(
                llm, system, [{"role": "user", "content": task}], Proposal, "tilesearch"
            )
        except Exception as e:
            log.warning("%s: proposal failed in round %d: %s", shape.label, round_index + 1, e)
            break
        result.usage.add(reply.usage)
        for config in proposal.configs[:per_round]:
            if config.label in tried:
                continue
            tried.add(config.label)
            trial = evaluate(config, shape, root, runner, smem)
            result.trials.append(trial)
            log.info("%s: %s", shape.label, trial.row)
    timed = [t for t in result.trials if t.us is not None]
    if not timed:
        return result
    result.best = min(timed, key=lambda t: t.us)
    best = root / f"{result.best.config.label}.py"
    spec = root / "spec.yaml"
    if fitted_trial.us is not None:
        baseline = root / f"{fitted.label}.py"
        result.vs_default = runner.compare(baseline, best, spec=spec, variant="bench-gpu").speedup
    comparison = runner.compare(reference, best, spec=spec, variant="bench-gpu")
    result.vs_cublas = comparison.speedup
    return result


def _task(
    shape: GemmShape,
    profile: DeviceProfile,
    result: SearchResult,
    per_round: int,
    round_index: int,
    rounds: int,
) -> str:
    d = profile.device
    lines = [
        f"GEMM: C[{shape.m}, {shape.n}] = A[{shape.m}, {shape.k}] @ "
        + (f"B[{shape.n}, {shape.k}].T" if shape.b_layout == "col" else f"B[{shape.k}, {shape.n}]")
        + f", {shape.dtype}, float accumulate.",
        f"Alignment: A {alignment(shape.k)}, B {alignment(shape.k if shape.b_layout == 'col' else shape.n)}, "
        f"C {alignment(shape.n)} elements per access.",
        f"Device: {d.name} ({d.arch}), {d.sm_count} SMs, {d.smem_per_block // 1024} KB shared "
        f"memory per block, {d.regs_per_sm} registers and {d.max_threads_per_sm} threads per SM.",
        f"cuBLAS: {result.cublas_us:.1f} us ({result.tflops(result.cublas_us):.1f} TFLOPS).",
        f"CUTLASS default {result.default.row}.",
    ]
    if result.fitted is not result.default:
        lines.append(f"Default tile with fewer stages: {result.fitted.row}.")
    if result.trials:
        lines.append("Configs tried so far:")
        lines += [f"- {t.row}" for t in result.trials]
    lines.append(
        f"Round {round_index + 1} of {rounds}: propose up to {per_round} new configs, "
        "different from those tried."
    )
    return "\n".join(lines)
