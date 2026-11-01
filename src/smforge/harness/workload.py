"""Loading kernel files and building their models and inputs on the GPU (worker side)."""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn

from smforge.spec import FLOAT_DTYPES, KernelSpec, Variant, load_spec

DTYPES = {
    name: getattr(torch, name)
    for name in (
        "float64",
        "float32",
        "float16",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint8",
        "bool",
    )
}
CASTABLE = frozenset({torch.float64, torch.float32, torch.float16, torch.bfloat16})
INIT_SEED = 42
INPUT_SEED = 123

_module_ids = itertools.count()


def load_module(path: str | Path) -> ModuleType:
    """Import a kernel file under a fresh module name, so two loads never share globals."""
    path = Path(path).resolve()
    name = f"smforge_kernel_{path.stem}_{next(_module_ids)}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_tensor(
    shape: tuple[int, ...],
    dtype: str,
    bounds: tuple[float, float] | None,
    device: torch.device,
) -> torch.Tensor:
    torch_dtype = DTYPES[dtype]
    if dtype == "bool":
        return torch.randint(0, 2, shape, device=device).bool()
    if dtype not in FLOAT_DTYPES:
        low, high = bounds or (0, 10)
        return torch.randint(int(low), int(high), shape, device=device, dtype=torch_dtype)
    if bounds is not None:
        values = torch.empty(shape, device=device, dtype=torch.float32).uniform_(*bounds)
        return values.to(torch_dtype)
    if dtype.startswith("float8"):
        return torch.randn(shape, device=device).to(torch_dtype)
    return torch.randn(shape, device=device, dtype=torch_dtype)


def clone(values: list) -> list:
    return [v.clone() if isinstance(v, torch.Tensor) else v for v in values]


def cuda_healthy() -> bool:
    """False once a CUDA error has left the context unusable."""
    try:
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


@dataclass
class Workload:
    """Inputs and ``Model`` construction for one spec variant.

    Without a spec, ``module``'s ``get_inputs`` and ``get_init_inputs`` decide, as in KernelBench.
    """

    device: torch.device
    spec: KernelSpec | None = None
    variant: Variant | None = None
    module: ModuleType | None = None

    @classmethod
    def create(
        cls,
        device: torch.device,
        spec: str | None,
        variant: str | None,
        index: int,
        fallback: str,
        dtype: str | None = None,
    ) -> Workload:
        """``dtype``, when given, replaces the variant's dtype."""
        if spec:
            kernel_spec = load_spec(spec)
            entry = kernel_spec.variant(variant, index)
            if dtype is not None:
                entry = entry.model_copy(update={"dtype": dtype})
            return cls(device, kernel_spec, entry)
        return cls(device, module=load_module(fallback))

    @property
    def dtype(self) -> torch.dtype | None:
        if self.spec is None:
            return None
        return DTYPES[self.spec.dtype(self.variant)]

    @property
    def flop(self) -> float | None:
        return self.spec.flop(self.variant) if self.spec else None

    @property
    def bytes_moved(self) -> float | None:
        return self.spec.bytes_moved(self.variant) if self.spec else None

    def init_args(self) -> list:
        if self.spec is not None:
            return self.spec.init_args(self.variant)
        get = getattr(self.module, "get_init_inputs", None)
        return list(get()) if get else []

    def inputs(self, draw: int = 0) -> list:
        set_seed(INPUT_SEED + draw)
        if self.spec is None:
            return [
                v.to(self.device) if isinstance(v, torch.Tensor) else v
                for v in self.module.get_inputs()
            ]
        return [
            make_tensor(shape, dtype, bounds, self.device)
            for shape, dtype, bounds in zip(
                self.spec.input_shapes(self.variant),
                self.spec.input_dtypes(self.variant),
                self.spec.input_ranges(self.variant),
                strict=True,
            )
        ]

    def build(self, module: ModuleType, dtype: torch.dtype | None = None) -> nn.Module:
        """``Model`` on the device, cast parameter by parameter so no full float32 copy lands."""
        set_seed(INIT_SEED)
        model = module.Model(*self.init_args()).eval()
        dtype = dtype or self.dtype
        if dtype in CASTABLE:
            return model.to(device=self.device, dtype=dtype)
        return model.to(self.device)

    def key(self) -> str:
        """Identifies the inputs and init arguments, for caching reference outputs."""
        if self.spec is None:
            return "module"
        inputs = {p: self.spec.inputs[p].model_dump() for p in self.variant.params}
        return json.dumps(
            {"variant": self.variant.model_dump(), "inputs": inputs, "inits": self.spec.inits},
            sort_keys=True,
        )
