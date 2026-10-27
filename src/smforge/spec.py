"""Kernel spec files: input shapes, ``Model.__init__`` arguments and benchmark variants.

The format is Xe-Forge's: an ``inputs`` section, an ``inits`` list, and a list of variants under
every other top-level key (``ci``, ``bench-gpu``, ``bench-gpu-1``, ...).
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

FLOAT_DTYPES = frozenset(
    {"float64", "float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"}
)
DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}
DEFAULT_VARIANT = "bench-gpu"
_RESERVED = frozenset({"inputs", "inits", "default_variant", "audit"})

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_FUNCTIONS = {"min": min, "max": max}


class InputSpec(BaseModel):
    shape: list[str | int]
    dtype: str = "float32"
    range: tuple[float | str, float | str] | None = None


class Variant(BaseModel):
    params: list[str]
    dims: dict[str, int] = Field(default_factory=dict)
    dtype: str | None = None
    flop: str | float | None = None
    bytes: str | float | None = None
    rtol: float | None = None
    atol: float | None = None


class AuditSpec(BaseModel):
    allow: list[str] = Field(default_factory=list)


class KernelSpec(BaseModel):
    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    inits: list[dict[str, Any]] = Field(default_factory=list)
    variants: dict[str, list[Variant]] = Field(default_factory=dict)
    default_variant: str | None = None
    audit: AuditSpec = Field(default_factory=AuditSpec)

    def resolve(self, name: str | None = None) -> str:
        name = name or self.default_variant or DEFAULT_VARIANT
        if name not in self.variants:
            raise KeyError(f"unknown variant {name!r}; the spec has {sorted(self.variants)}")
        return name

    def variant(self, name: str | None = None, index: int = 0) -> Variant:
        entries = self.variants[self.resolve(name)]
        if not 0 <= index < len(entries):
            raise IndexError(f"variant {name!r} has {len(entries)} entries, not {index + 1}")
        return entries[index]

    def input_shapes(self, variant: Variant) -> list[tuple[int, ...]]:
        return [
            tuple(_dim(d, variant.dims) for d in self.inputs[param].shape)
            for param in variant.params
        ]

    def input_ranges(self, variant: Variant) -> list[tuple[float, float] | None]:
        ranges = []
        for param in variant.params:
            bounds = self.inputs[param].range
            ranges.append(tuple(_dim(b, variant.dims) for b in bounds) if bounds else None)
        return ranges

    def input_dtypes(self, variant: Variant) -> list[str]:
        """Each input's dtype. A variant dtype overrides floating inputs only."""
        dtypes = []
        for param in variant.params:
            declared = self.inputs[param].dtype
            override = variant.dtype and declared in FLOAT_DTYPES
            dtypes.append(variant.dtype if override else declared)
        return dtypes

    def dtype(self, variant: Variant) -> str:
        if variant.dtype:
            return variant.dtype
        floats = [d for d in self.input_dtypes(variant) if d in FLOAT_DTYPES]
        return floats[0] if floats else "float32"

    def init_args(self, variant: Variant) -> list:
        """``Model.__init__`` arguments. Dim names resolve, also inside lists, which become tuples."""
        return [_init_value(v, variant.dims) for entry in self.inits for v in entry.values()]

    def flop(self, variant: Variant) -> float | None:
        return evaluate(variant.flop, variant.dims)

    def bytes_moved(self, variant: Variant) -> float | None:
        return evaluate(variant.bytes, variant.dims)


class _SpecDumper(yaml.SafeDumper):
    def ignore_aliases(self, data) -> bool:
        return True


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    flow = all(not isinstance(item, list | dict) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_SpecDumper.add_representer(list, _represent_list)


def dump_spec(document: dict) -> str:
    """YAML for a spec, with shapes and other flat lists on one line."""
    return yaml.dump(document, Dumper=_SpecDumper, sort_keys=False)


def load_spec(path: str | Path) -> KernelSpec:
    return parse_spec(yaml.safe_load(Path(path).read_text()))


def parse_spec(data: dict) -> KernelSpec:
    variants = {
        key: [Variant.model_validate(entry) for entry in value]
        for key, value in data.items()
        if key not in _RESERVED and isinstance(value, list)
    }
    return KernelSpec(
        inputs=data.get("inputs") or {},
        inits=data.get("inits") or [],
        variants=variants,
        default_variant=data.get("default_variant"),
        audit=data.get("audit") or {},
    )


def evaluate(expr: str | float | None, dims: dict[str, int]) -> float | None:
    """Evaluate a spec formula such as ``"2*M*N*K"`` with the variant's dims."""
    if expr is None:
        return None
    if isinstance(expr, int | float):
        return float(expr)
    return float(_eval(ast.parse(str(expr), mode="eval").body, dims))


def _eval(node: ast.expr, dims: dict[str, int]) -> float:
    match node:
        case ast.Constant(value=int() | float() as value):
            return value
        case ast.Name(id=name):
            if name not in dims:
                raise ValueError(f"unknown dim {name!r} in formula")
            return dims[name]
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINOPS:
            return _BINOPS[type(op)](_eval(left, dims), _eval(right, dims))
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval(operand, dims)
        case ast.Call(func=ast.Name(id=name), args=args) if name in _FUNCTIONS:
            return _FUNCTIONS[name](*(_eval(a, dims) for a in args))
    raise ValueError(f"unsupported formula expression: {ast.unparse(node)}")


def _init_value(value: Any, dims: dict[str, int]) -> Any:
    if isinstance(value, str):
        return dims.get(value, value)
    if isinstance(value, list):
        return tuple(_init_value(v, dims) for v in value)
    return value


def _dim(value: str | float, dims: dict[str, int]) -> float:
    if isinstance(value, int | float):
        return value
    if value not in dims:
        raise ValueError(f"shape uses dim {value!r}, which the variant does not set")
    return dims[value]
