"""Device properties from torch, the driver and ``nvidia-smi`` (worker side)."""

from __future__ import annotations

import torch
import triton

from smforge.device.tools import is_wsl, nvidia_smi
from smforge.models import DeviceInfo


def query_device(index: int = 0) -> DeviceInfo:
    props = torch.cuda.get_device_properties(index)
    smi = nvidia_smi(index, "driver_version", "clocks.max.sm", "clocks.max.memory", "power.limit")
    return DeviceInfo(
        index=index,
        name=props.name,
        capability=(props.major, props.minor),
        sm_count=props.multi_processor_count,
        memory_bytes=props.total_memory,
        l2_bytes=props.L2_cache_size,
        smem_per_block=props.shared_memory_per_block_optin,
        smem_per_sm=props.shared_memory_per_multiprocessor,
        regs_per_sm=props.regs_per_multiprocessor,
        max_threads_per_sm=props.max_threads_per_multi_processor,
        max_ctas_per_sm=_max_ctas_per_sm(index),
        warp_size=props.warp_size,
        memory_bus_bits=props.memory_bus_width,
        max_sm_clock_mhz=_number(smi.get("clocks.max.sm"), int),
        max_memory_clock_mhz=_number(smi.get("clocks.max.memory"), int),
        power_limit_w=_number(smi.get("power.limit"), float),
        driver=smi.get("driver_version"),
        cuda=torch.version.cuda,
        torch=torch.__version__,
        triton=triton.__version__,
        wsl=is_wsl(),
    )


def _max_ctas_per_sm(index: int) -> int | None:
    try:
        from cuda.bindings import driver
    except ImportError:
        return None
    attribute = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR
    err, value = driver.cuDeviceGetAttribute(attribute, index)
    return value if err == driver.CUresult.CUDA_SUCCESS else None


def _number(value: str | None, cast: type) -> int | float | None:
    try:
        return cast(float(value)) if value is not None else None
    except ValueError:
        return None
