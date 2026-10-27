"""Device profiles: ``hardware/<gpu>.yaml``, written by ``smforge device --save``."""

from __future__ import annotations

from pathlib import Path

import yaml

from smforge.device.tools import nvidia_smi
from smforge.models import DeviceInfo, DeviceProfile


def profile_path(hardware_dir: Path, device: DeviceInfo) -> Path:
    return hardware_dir / f"{device.slug}.yaml"


def load_profile(path: Path) -> DeviceProfile:
    return DeviceProfile.model_validate(yaml.safe_load(path.read_text()))


def save_profile(profile: DeviceProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False))


def find_profile(hardware_dir: Path, index: int = 0) -> DeviceProfile | None:
    """The saved profile for the GPU at ``index``, matched by name, without starting CUDA."""
    name = nvidia_smi(index, "name").get("name")
    if name is None or not hardware_dir.is_dir():
        return None
    for path in sorted(hardware_dir.glob("*.yaml")):
        profile = load_profile(path)
        if profile.device.name == name:
            return profile
    return None
