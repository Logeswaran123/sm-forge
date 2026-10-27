"""smforge: device baseline now; the optimization pipeline in later stages."""

from __future__ import annotations

import argparse
import logging
import sys

from smforge import render
from smforge.config import get_config
from smforge.device.profile import load_profile, profile_path, save_profile
from smforge.device.tools import check_tools
from smforge.harness.runner import Runner, WorkerError
from smforge.models import DeviceInfo, DeviceProfile, NoiseResult, ProbeResult, Roofline


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smforge", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    device = commands.add_parser("device", help="query the GPU and measure its baseline (stage 0)")
    device.add_argument("--probes", action="store_true", help="compile feature probes")
    device.add_argument("--roofline", action="store_true", help="measure bandwidth and TFLOPS")
    device.add_argument("--noise", action="store_true", help="A/A timing noise")
    device.add_argument("--tools", action="store_true", help="check ncu, nsys, compute-sanitizer")
    device.add_argument("--all", action="store_true", help="everything above, then --save")
    device.add_argument("--save", action="store_true", help="write hardware/<gpu>.yaml")
    device.add_argument("--json", action="store_true", help="print JSON")
    device.set_defaults(func=cmd_device)
    return parser


def cmd_device(args: argparse.Namespace) -> int:
    config = get_config()
    with Runner(config) as runner:
        info = DeviceInfo.model_validate(runner.call("device"))
        path = profile_path(config.hardware_dir, info)
        profile = load_profile(path) if path.exists() else DeviceProfile(device=info)
        profile.device = info
        if args.probes or args.all:
            profile.probes = [_probe(runner, name) for name in runner.call("probe_names")]
        if args.roofline or args.all:
            profile.roofline = Roofline.model_validate(runner.call("roofline"))
        if args.noise or args.all:
            noise = runner.call("noise", timing=config.timing.model_dump())
            profile.noise = NoiseResult.model_validate(noise)
    if args.tools or args.all:
        profile.tools = check_tools()
    if args.save or args.all:
        save_profile(profile, path)
        print(f"wrote {path}", file=sys.stderr)
    print(profile.model_dump_json(indent=2) if args.json else render.device(profile))
    return 0


def _probe(runner: Runner, name: str) -> ProbeResult:
    try:
        return ProbeResult.model_validate(runner.call("probe", name=name))
    except WorkerError as e:
        return ProbeResult(name=name, compiled=False, error=str(e))


if __name__ == "__main__":
    sys.exit(main())
