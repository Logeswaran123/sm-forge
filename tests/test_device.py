import pytest

from smforge.config import Config, TimingConfig
from smforge.device.profile import find_profile, load_profile, profile_path, save_profile
from smforge.device.tools import nvidia_smi
from smforge.harness.runner import Runner
from smforge.models import (
    DeviceInfo,
    DeviceProfile,
    NoiseResult,
    ProbeResult,
    Roofline,
)

DEVICE = DeviceInfo(
    index=0,
    name="NVIDIA GeForce RTX 5080",
    capability=(12, 0),
    sm_count=84,
    memory_bytes=16 << 30,
    l2_bytes=64 << 20,
    smem_per_block=101376,
    smem_per_sm=102400,
    regs_per_sm=65536,
    max_threads_per_sm=1536,
    warp_size=32,
    memory_bus_bits=256,
    torch="2.11.0",
    triton="3.6.0",
)


def test_profile_roundtrip(tmp_path):
    profile = DeviceProfile(
        device=DEVICE,
        probes=[
            ProbeResult(name="mma_fp16", compiled=True, lowered=True, correct=True),
            ProbeResult(name="warp_specialize", compiled=False, error="RuntimeError"),
        ],
        roofline=Roofline(
            dram_copy_gbps=800, dram_read_gbps=900, l2_copy_gbps=3000, tflops={"float16": 180}
        ),
    )
    path = profile_path(tmp_path, DEVICE)
    assert path.name == "rtx5080.yaml"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded == profile
    assert loaded.device.arch == "sm_120"
    assert loaded.features == {"mma_fp16": True, "warp_specialize": False}
    assert loaded.roofline.ridge("float16") == pytest.approx(200)
    assert loaded.roofline.ridge("float64") is None


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    config = Config(
        cache_dir=tmp_path_factory.mktemp("cache"),
        timing=TimingConfig(warmup=5, iters=20, rounds=2, flush_mb=64),
    )
    with Runner(config) as runner:
        yield runner


@pytest.mark.gpu
def test_device_query(runner, tmp_path):
    info = DeviceInfo.model_validate(runner.call("device"))
    assert info.sm_count > 0
    assert info.smem_per_block >= 48 * 1024
    assert info.arch.startswith("sm_")
    assert info.max_ctas_per_sm
    assert info.name == nvidia_smi(0, "name")["name"]
    save_profile(DeviceProfile(device=info), profile_path(tmp_path, info))
    assert find_profile(tmp_path).device == info


@pytest.mark.gpu
def test_probes(runner):
    names = runner.call("probe_names")
    assert {"mma_fp16", "tma_host", "warp_specialize"} <= set(names)
    fp16 = ProbeResult.model_validate(runner.call("probe", name="mma_fp16"))
    assert fp16.supported
    assert any("MMA" in op for op in fp16.instructions)


@pytest.mark.gpu
def test_noise(runner):
    noise = NoiseResult.model_validate(
        runner.call("noise", timing=runner.config.timing.model_dump())
    )
    assert [case.name for case in noise.cases] == ["launch_bound", "memory_bound", "compute_bound"]
    assert noise.min_effect >= 0.01
