"""smforge-skill: harness commands for one kernel file.

validate  static checks and the forward-path audit, no GPU
check     correctness against an oracle, then the launch audit
bench     device time, or an interleaved comparison with --baseline
facts     registers, spills, shared memory, SASS mix and occupancy per launched kernel
sanitize  compute-sanitizer memcheck, racecheck, initcheck or synccheck
trial     the trial tree: init, save (check against t0), status, best, finalize
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import yaml

from smforge import render
from smforge.config import get_config
from smforge.corpus.catalog import Problem, load_problems
from smforge.device.profile import find_profile
from smforge.harness.audit import audit_launches, audit_source
from smforge.harness.runner import Runner
from smforge.harness.sanitizer import TOOLS, SanitizerError, sanitize
from smforge.harness.static import validate_source
from smforge.models import Severity
from smforge.spec import DTYPE_BYTES, KernelSpec, load_spec
from smforge.trials import TrialError, TrialTree


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except TrialError as e:
        print(e, file=sys.stderr)
        return 2


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

    sanitize = command(
        "sanitize", "compute-sanitizer on a warm-up forward and one more", cmd_sanitize
    )
    sanitize.add_argument("--tool", choices=TOOLS, default="memcheck")

    trial = commands.add_parser("trial", help="the trial tree: init, save, status, best, finalize")
    trial_commands = trial.add_subparsers(dest="trial_command", required=True)

    def trial_command(name: str, help: str, func: Callable[[argparse.Namespace], int]):
        sub = trial_commands.add_parser(name, help=help)
        sub.add_argument("--dir", type=Path, default=Path("trials"), help="trial tree directory")
        sub.set_defaults(func=func)
        return sub

    init = trial_command("init", "measure the baseline and store it as t0", cmd_trial_init)
    init.add_argument("--problem", required=True, help="corpus problem id, or a problem directory")
    init.add_argument("--baseline", type=Path, help="baseline kernel (default: the problem's seed)")
    save = trial_command(
        "save", "store a kernel as a trial and check it against t0", cmd_trial_save
    )
    save.add_argument("kernel", type=Path)
    save.add_argument("--parent", default="t0", help="the trial this one was derived from")
    save.add_argument("--strategy", required=True, help="what this trial changes, in one line")
    trial_command("status", "the tree with each trial's outcome", cmd_trial_status)
    trial_command("best", "the best trial so far", cmd_trial_best)
    finalize = trial_command("finalize", "copy the best faster trial to a file", cmd_trial_finalize)
    finalize.add_argument("output", type=Path)
    finalize.add_argument("--no-memcheck", action="store_true", help="skip compute-sanitizer")
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


def cmd_sanitize(args: argparse.Namespace) -> int:
    try:
        report = sanitize(args.kernel, args.tool, args.spec, args.variant)
    except SanitizerError as e:
        print(f"compute-sanitizer unavailable: {e}", file=sys.stderr)
        return 2
    print(report.model_dump_json(indent=2) if args.json else render.sanitizer(report))
    return 0 if report.ok else 1


def cmd_trial_init(args: argparse.Namespace) -> int:
    problem = find_problem(args.problem)
    baseline = args.baseline or problem.seed
    with Runner() as runner:
        tree = TrialTree.create(args.dir, problem, baseline, runner)
    print(tree.status())
    return 0


def cmd_trial_save(args: argparse.Namespace) -> int:
    tree = TrialTree(args.dir)
    trial = tree.save(args.kernel, args.parent, args.strategy)
    with Runner() as runner:
        outcome = tree.evaluate(trial, runner)
    head = f"{trial.id} (from {trial.parent}): {outcome.kind}"
    if trial.speedup is not None:
        head += (
            f", {trial.speedup:.3f}x over t0 ({trial.baseline_us:.1f} -> {trial.time_us:.1f} us)"
        )
    print(f"{head}\n\n{outcome.feedback}")
    return 0 if trial.correct else 1


def cmd_trial_status(args: argparse.Namespace) -> int:
    print(TrialTree(args.dir).status())
    return 0


def cmd_trial_best(args: argparse.Namespace) -> int:
    tree = TrialTree(args.dir)
    best = tree.best()
    if best is None:
        print("no correct trial yet")
        return 1
    print(f"{best.id}: {best.outcome}, {best.speedup:.3f}x over t0 | {best.strategy}")
    print(tree.path(best))
    return 0


def cmd_trial_finalize(args: argparse.Namespace) -> int:
    tree = TrialTree(args.dir)
    spec = load_spec(tree.problem.spec)
    variant = "ci" if "ci" in spec.variants else None

    def memcheck(kernel: Path):
        return sanitize(kernel, "memcheck", tree.problem.spec, variant)

    try:
        final = tree.finalize(args.output, None if args.no_memcheck else memcheck)
    except SanitizerError as e:
        print(f"compute-sanitizer unavailable: {e}; rerun with --no-memcheck", file=sys.stderr)
        return 2
    if final is None:
        print("nothing finalized: no trial is faster than t0 and memcheck-clean")
        print(tree.status())
        return 1
    print(f"finalized {final.id}, {final.speedup:.3f}x over t0 -> {args.output}")
    return 0


def find_problem(name: str) -> Problem:
    """A corpus problem by id, or a directory with ``reference.py`` and ``spec.yaml``."""
    path = Path(name)
    if (path / "spec.yaml").exists():
        meta = path / "problem.yaml"
        data = yaml.safe_load(meta.read_text()) if meta.exists() else {}
        data.setdefault("id", path.resolve().name)
        data.setdefault("source", "local")
        data.setdefault("family", "unknown")
        return Problem(**data, path=path)
    found = [p for p in load_problems() if p.id == name]
    if not found:
        raise SystemExit(f"no problem {name!r} in the corpus, and no spec.yaml in {name}")
    return found[0]


def _workload(args: argparse.Namespace) -> dict:
    return {"spec": args.spec, "variant": args.variant, "index": args.index}


def _dtype(spec: KernelSpec, args: argparse.Namespace) -> str:
    return spec.dtype(spec.variant(args.variant, args.index))


if __name__ == "__main__":
    sys.exit(main())
