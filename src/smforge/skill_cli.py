"""smforge-skill: harness commands for one kernel file.

validate  static checks and the forward-path audit, no GPU
check     correctness against an oracle, then the launch audit
bench     device time, or an interleaved comparison with --baseline
facts     registers, spills, shared memory, SASS mix and occupancy per launched kernel
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from smforge import render
from smforge.config import get_config
from smforge.device.profile import find_profile
from smforge.harness.audit import audit_launches, audit_source
from smforge.harness.runner import Runner
from smforge.harness.static import validate_source
from smforge.models import Severity
from smforge.spec import DTYPE_BYTES, KernelSpec, load_spec


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smforge-skill",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str, func: Callable[[argparse.Namespace], int]):
        sub = commands.add_parser(name, help=help)
        sub.add_argument("kernel", type=Path)
        sub.add_argument("--spec", type=Path, help="spec.yaml with shapes and variants")
        sub.add_argument("--variant", help="spec variant (default: the spec's, else bench-gpu)")
        sub.add_argument("--index", type=int, default=0, help="entry within the variant")
        sub.add_argument("--json", action="store_true", help="print JSON")
        sub.set_defaults(func=func)
        return sub

    command("validate", "static checks and the forward-path audit, no GPU", cmd_validate)

    check = command("check", "correctness against an oracle, then the launch audit", cmd_check)
    check.add_argument("--reference", type=Path, help="PyTorch reference; the oracle if given")
    check.add_argument(
        "--baseline",
        type=Path,
        help="baseline kernel: the oracle when there is no reference, and the launch-audit baseline",
    )
    check.add_argument("--rtol", type=float)
    check.add_argument("--atol", type=float)
    check.add_argument("--draws", type=int, help="input draws to compare")

    bench = command("bench", "device time, or a comparison with --baseline", cmd_bench)
    bench.add_argument("--baseline", type=Path, help="time against this kernel, interleaved")
    bench.add_argument("--graph", action="store_true", help="replay the forward as a CUDA graph")

    command("facts", "per-kernel resources, SASS mix and occupancy", cmd_facts)
    return parser


def cmd_validate(args: argparse.Namespace) -> int:
    code = args.kernel.read_text()
    spec = load_spec(args.spec) if args.spec else None
    config = get_config()
    profile = find_profile(config.hardware_dir, config.device)
    issues = validate_source(
        code,
        smem_limit=profile.device.smem_per_block if profile else None,
        elem_bytes=DTYPE_BYTES.get(_dtype(spec, args)) if spec else None,
    )
    if not any(issue.check == "syntax_error" for issue in issues):
        issues += audit_source(code, spec.audit.allow if spec else ())
    if args.json:
        print(json.dumps([issue.model_dump(mode="json") for issue in issues], indent=2))
    else:
        print(render.issues(issues))
    return 1 if any(issue.severity is Severity.ERROR for issue in issues) else 0


def cmd_check(args: argparse.Namespace) -> int:
    oracle = args.reference or args.baseline
    if oracle is None:
        print("check needs --reference or --baseline", file=sys.stderr)
        return 2
    workload = _workload(args)
    issues, launches = [], None
    with Runner() as runner:
        result = runner.check(
            args.kernel,
            oracle,
            rtol=args.rtol,
            atol=args.atol,
            draws=args.draws,
            **workload,
        )
        if result.passed:
            launches = runner.launches(args.kernel, **workload)
            baseline = runner.launches(args.baseline, **workload) if args.baseline else None
            issues = audit_launches(launches, baseline)
    if args.json:
        print(
            json.dumps(
                {
                    "check": result.model_dump(mode="json"),
                    "launches": launches.model_dump(mode="json") if launches else None,
                    "audit": [issue.model_dump(mode="json") for issue in issues],
                },
                indent=2,
            )
        )
    else:
        print(render.check(result))
        if launches is not None:
            print(render.launches(launches))
            print(f"launch audit: {render.issues(issues)}")
    failed = not result.passed or any(issue.severity is Severity.ERROR for issue in issues)
    return 1 if failed else 0


def cmd_bench(args: argparse.Namespace) -> int:
    with Runner() as runner:
        if args.baseline:
            result = runner.compare(args.baseline, args.kernel, graph=args.graph, **_workload(args))
            text = render.comparison(result)
        else:
            result = runner.bench(args.kernel, graph=args.graph, **_workload(args))
            text = render.timing(result)
    print(result.model_dump_json(indent=2) if args.json else text)
    return 0


def cmd_facts(args: argparse.Namespace) -> int:
    with Runner() as runner:
        report = runner.facts(args.kernel, **_workload(args))
    if args.json:
        print(report.model_dump_json(indent=2))
        return 0
    ridge = None
    config = get_config()
    profile = find_profile(config.hardware_dir, config.device)
    if profile and profile.roofline and args.spec:
        ridge = profile.roofline.ridge(_dtype(load_spec(args.spec), args))
    print(render.facts(report, ridge))
    return 0


def _workload(args: argparse.Namespace) -> dict:
    return {"spec": args.spec, "variant": args.variant, "index": args.index}


def _dtype(spec: KernelSpec, args: argparse.Namespace) -> str:
    return spec.dtype(spec.variant(args.variant, args.index))


if __name__ == "__main__":
    sys.exit(main())
