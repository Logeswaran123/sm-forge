"""What Nsight Compute profiles: one warm-up forward, then one forward inside the profiler range."""

from __future__ import annotations

import argparse

import torch

from smforge.config import get_config
from smforge.harness.workload import Workload, load_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("--spec")
    parser.add_argument("--variant")
    args = parser.parse_args()
    device = torch.device("cuda", get_config().device)
    torch.cuda.set_device(device)
    workload = Workload.create(device, args.spec, args.variant, 0, args.kernel)
    model = workload.build(load_module(args.kernel))
    inputs = workload.inputs()
    with torch.no_grad():
        model(*inputs)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        model(*inputs)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
