import shutil
import subprocess

import pytest


def _has_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def pytest_collection_modifyitems(config, items):
    if _has_gpu():
        return
    skip = pytest.mark.skip(reason="no NVIDIA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
