"""Importing KernelBench problems and fitting them to the GPU.

The spec keeps KernelBench's shapes as the ``kernelbench`` variant. ``bench-gpu`` halves one dim,
the batch or else the leading dim of the first input, until the float16 reference fits
``PEAK_BYTES`` and ``TIME_US``; a reference that cannot run at a size (out of memory) counts as
over. It then takes the first of float16, bfloat16 and float32 whose reference result is finite
and within that dtype's tolerance of float32. ``ci`` is ``bench-gpu`` with that dim at
``CI_BATCH``, a shape that exercises masking and is never used while optimizing.

Each problem is fitted in a fresh worker. On WSL2 an out-of-memory attempt can spill into system
memory, and a worker that did that measures the next problem several times too slow.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from smforge.corpus.catalog import Problem, save_problem
from smforge.corpus.kernelbench import family, problem_id, translate
from smforge.harness.runner import Runner, WorkerError
from smforge.models import ReferenceProfile
from smforge.spec import dump_spec

PEAK_BYTES = 4 << 30
TIME_US = 5000.0
CI_BATCH = 3
BATCH_DIM = "BATCH_SIZE"
DTYPES = ("float16", "bfloat16", "float32")


def import_kernelbench(
    path: Path, level: int, root: Path, runner: Runner, upstream: str
) -> Problem:
    sections, dims = translate(path)
    runner.close()
    pid = problem_id(level, path)
    problem = Problem(
        id=pid,
        source="kernelbench",
        family=family(level, path),
        upstream=upstream,
        path=root / "kernelbench" / pid,
    )
    problem.path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, problem.reference)
    params = list(sections["inputs"])
    upstream_variant = {"params": params, "dims": dims}
    write_spec(problem, sections, {"kernelbench": [upstream_variant]})
    runner.call(
        "verify_translation",
        reference=str(problem.reference),
        spec=str(problem.spec),
        variant="kernelbench",
    )
    dtype, fitted, profile, problem.notes = _fit(problem, sections, params, dims, runner)
    variants = {"bench-gpu": [_variant(params, dtype, fitted, profile)]}
    ci = _ci(problem, sections, params, dtype, fitted, runner)
    if ci is not None:
        variants["ci"] = [ci]
    variants["kernelbench"] = [upstream_variant]
    write_spec(problem, sections, variants)
    save_problem(problem)
    return problem


def write_spec(problem: Problem, sections: dict, variants: dict[str, list[dict]]) -> None:
    document = {**sections, "default_variant": next(iter(variants)), **variants}
    problem.spec.write_text(dump_spec(document))


def scale_dim(sections: dict, dims: dict[str, int]) -> str | None:
    """The dim to shrink: the batch, else the first input's leading dim if no init uses it."""
    if BATCH_DIM in dims:
        return BATCH_DIM
    shape = next(iter(sections["inputs"].values()))["shape"]
    lead = shape[0] if shape and isinstance(shape[0], str) else None
    used = {str(v) for entry in sections["inits"] for v in _flatten(list(entry.values()))}
    return lead if lead in dims and lead not in used else None


def _flatten(values: list) -> list:
    return [x for v in values for x in (_flatten(v) if isinstance(v, list) else [v])]


def _fit(
    problem: Problem, sections: dict, params: list[str], dims: dict[str, int], runner: Runner
) -> tuple[str, dict[str, int], ReferenceProfile, list[str]]:
    fitted, notes = dict(dims), []
    scaled = scale_dim(sections, dims)
    budget = f"{PEAK_BYTES >> 30} GiB and {TIME_US / 1000:g} ms"
    reasons = []
    while True:
        try:
            profile = _profile(problem, sections, params, "float16", fitted, runner)
            over = profile.peak_bytes > PEAK_BYTES or profile.time_us > TIME_US
            reason = f"{profile.peak_bytes / 2**30:.1f} GiB, {profile.time_us / 1000:.1f} ms"
        except WorkerError as e:
            if scaled is None or fitted[scaled] == 1:
                raise
            profile, over, reason = None, True, str(e).splitlines()[0][:80]
        if not over or scaled is None or fitted[scaled] == 1:
            break
        reasons.append(f"{fitted[scaled]}: {reason}")
        fitted[scaled] = max(1, fitted[scaled] // 2)
    if scaled and fitted[scaled] != dims[scaled]:
        notes.append(
            f"{scaled} {dims[scaled]} -> {fitted[scaled]} to fit {budget} in float16 "
            f"({'; '.join(reasons)})"
        )
    elif over:
        notes.append(
            f"over {budget} at KernelBench shapes: {profile.peak_bytes / 2**30:.1f} GiB, "
            f"{profile.time_us / 1000:.1f} ms"
        )
    for dtype in DTYPES:
        if dtype != "float16":
            profile = _profile(problem, sections, params, dtype, fitted, runner)
        if profile.finite and profile.stable is not False:
            return dtype, fitted, profile, notes
        if not profile.finite:
            notes.append(f"{dtype} reference output is not finite")
        else:
            notes.append(
                f"{dtype} reference differs from float32 by up to {profile.max_abs_error:.3g}"
            )
    return "float32", fitted, profile, notes


def _ci(
    problem: Problem,
    sections: dict,
    params: list[str],
    dtype: str,
    dims: dict[str, int],
    runner: Runner,
) -> dict | None:
    scaled = scale_dim(sections, dims)
    if scaled is None or dims[scaled] <= CI_BATCH:
        return None
    ci_dims = {**dims, scaled: CI_BATCH}
    try:
        profile = _profile(problem, sections, params, dtype, ci_dims, runner)
    except WorkerError as e:
        problem.notes.append(f"no ci variant: {e}")
        return None
    if not profile.finite:
        problem.notes.append("no ci variant: reference output is not finite")
        return None
    return _variant(params, dtype, ci_dims, profile)


def _profile(
    problem: Problem,
    sections: dict,
    params: list[str],
    dtype: str,
    dims: dict[str, int],
    runner: Runner,
) -> ReferenceProfile:
    write_spec(problem, sections, {"bench-gpu": [{"params": params, "dtype": dtype, "dims": dims}]})
    return runner.profile_reference(problem.reference, spec=problem.spec, variant="bench-gpu")


def _variant(params: list[str], dtype: str, dims: dict[str, int], profile: ReferenceProfile):
    variant = {"params": params, "dtype": dtype, "dims": dict(dims)}
    if profile.flop:
        variant["flop"] = int(profile.flop)
    variant["bytes"] = int(profile.bytes_moved)
    return variant
