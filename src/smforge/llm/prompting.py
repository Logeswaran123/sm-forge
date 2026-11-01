"""Prompt files per kernel language, and the device and toolchain values they are filled with.

Triton prompts sit in ``prompts/``, CUDA C++ prompts in ``prompts/cuda/`` with the same names;
``analyze.md`` and ``plan.md`` serve both. A CUDA prompt's ``$format`` is ``prompts/cuda/format.md``,
the kernel-file layout ``smforge.cuda.build`` expects.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from smforge.cuda import CUTLASS_VERSION
from smforge.models import DeviceProfile, Dsl

PROMPTS = Path(__file__).parent / "prompts"


def prompt_dir(dsl: Dsl) -> Path:
    return PROMPTS / "cuda" if dsl is Dsl.CUDA else PROMPTS


def stage_text(dsl: Dsl, stage: str) -> str:
    return (prompt_dir(dsl) / "stages" / f"{stage}.md").read_text().strip()


def values(profile: DeviceProfile | None, dsl: Dsl, features: bool = False) -> dict[str, str]:
    """``device``, ``triton``, ``cuda``, ``arch``, ``torch``, and for CUDA ``format``.

    ``features`` lists the device probes that passed in ``device``, as the seed prompt does.
    """
    if profile is None:
        found = {
            "device": "an NVIDIA GPU",
            "triton": "3.x",
            "torch": "2.x",
            "cuda": "12.x",
            "arch": "sm_80",
            "smem": "99",
        }
    else:
        d = profile.device
        if features:
            supported = [name for name, ok in profile.features.items() if ok]
            device = (
                f"{d.name} ({d.arch}), {d.sm_count} SMs, {d.smem_per_block // 1024} KB shared "
                f"memory per block; supported: {', '.join(supported)}"
            )
        else:
            device = (
                f"{d.name} ({d.arch}), {d.sm_count} SMs, {d.l2_bytes >> 20} MB L2, "
                f"{d.smem_per_block // 1024} KB shared memory per block"
            )
        major, minor = d.capability
        found = {
            "device": device,
            "triton": d.triton,
            "torch": d.torch.split("+")[0],
            "cuda": d.cuda or "12.x",
            "arch": f"sm_{major}{minor}" + ("a" if major >= 9 else ""),
            "smem": str(d.smem_per_block // 1024),
        }
    found["cutlass"] = CUTLASS_VERSION
    if dsl is Dsl.CUDA:
        found["format"] = Template((PROMPTS / "cuda" / "format.md").read_text().strip()).substitute(
            found
        )
    return found


def fill(dsl: Dsl, name: str, profile: DeviceProfile | None, **extra: str) -> str:
    """Prompt ``name`` for ``dsl``, filled with the device values and ``extra``."""
    template = Template((prompt_dir(dsl) / name).read_text())
    return template.substitute(values(profile, dsl, features=name == "seed.md") | extra)
