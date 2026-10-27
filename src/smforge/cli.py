"""smforge: device baseline and the problem corpus; the optimization pipeline in later stages."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from smforge import render
from smforge.config import get_config
from smforge.corpus.build import import_kernelbench
from smforge.corpus.catalog import (
    CORPUS_DIR,
    Problem,
    load_problems,
    load_splits,
    make_splits,
    save_splits,
)
from smforge.corpus.kernelbench import LEVEL1, Untranslatable, problem_id, problem_number
from smforge.corpus.results import provenance, upsert
from smforge.device.profile import load_profile, profile_path, save_profile
from smforge.device.tools import check_tools
from smforge.harness.runner import Runner, WorkerError
from smforge.models import DeviceInfo, DeviceProfile, NoiseResult, ProbeResult, Roofline
from smforge.spec import load_spec


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

    corpus = commands.add_parser("corpus", help="build and measure the problem corpus (stage 2)")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)

    kernelbench = corpus_commands.add_parser(
        "import-kernelbench", help="translate KernelBench problems into the corpus"
    )
    kernelbench.add_argument("root", type=Path, help="KernelBench checkout")
    kernelbench.add_argument("--level", type=int, choices=(1, 2), required=True)
    kernelbench.add_argument("--only", help="comma-separated problem numbers")
    kernelbench.add_argument("--force", action="store_true", help="re-import existing problems")
    kernelbench.set_defaults(func=cmd_import_kernelbench)

    listing = corpus_commands.add_parser("list", help="problems, their split and seed status")
    listing.add_argument("--split", choices=("dev", "eval"))
    listing.set_defaults(func=cmd_list)

    split = corpus_commands.add_parser("split", help="assign new problems to dev or eval")
    split.add_argument("--eval-fraction", type=float, default=0.3)
    split.add_argument("--seed", type=int, default=0)
    split.set_defaults(func=cmd_split)

    baselines = corpus_commands.add_parser(
        "baselines", help="time eager, torch.compile and seed kernels into results/baselines.csv"
    )
    baselines.add_argument("--ids", help="comma-separated problem ids")
    baselines.add_argument("--split", choices=("dev", "eval"))
    baselines.add_argument("--modes", default="eager,compile,seed")
    baselines.add_argument("--variant", default="bench-gpu")
    baselines.add_argument(
        "--compile-timeout", type=float, default=1800, help="seconds allowed per compile"
    )
    baselines.set_defaults(func=cmd_baselines)
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


def cmd_import_kernelbench(args: argparse.Namespace) -> int:
    level_dir = args.root / "KernelBench" / f"level{args.level}"
    commit = subprocess.run(
        ["git", "-C", str(args.root), "rev-parse", "--short=12", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    only = {int(n) for n in args.only.split(",")} if args.only else None
    paths = sorted(level_dir.glob("*.py"), key=problem_number)
    if args.level == 1:
        paths = [p for p in paths if problem_number(p) in LEVEL1]
    if only is not None:
        paths = [p for p in paths if problem_number(p) in only]
    root = CORPUS_DIR / "kernelbench"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.root / "LICENSE", root / "LICENSE")
    failures = 0
    with Runner() as runner:
        for path in paths:
            target = root / problem_id(args.level, path)
            if (target / "problem.yaml").exists() and not args.force:
                continue
            upstream = f"KernelBench/level{args.level}/{path.name}@{commit}"
            try:
                problem = import_kernelbench(path, args.level, CORPUS_DIR, runner, upstream)
            except (Untranslatable, WorkerError, ValueError) as e:
                failures += 1
                print(f"{problem_id(args.level, path)}: FAILED {e}", flush=True)
                continue
            spec = load_spec(problem.spec)
            variant = spec.variant()
            notes = f"  ({'; '.join(problem.notes)})" if problem.notes else ""
            print(f"{problem.id}: {spec.dtype(variant)} {variant.dims}{notes}", flush=True)
    return 1 if failures else 0


def cmd_list(args: argparse.Namespace) -> int:
    splits = load_splits()
    split_of = {pid: name for name, ids in splits.items() for pid in ids}
    for problem in load_problems():
        split = split_of.get(problem.id, "-")
        if args.split and split != args.split:
            continue
        seeded = "seeded" if problem.seed.exists() else "-"
        print(f"{problem.id:<60}{problem.family:<17}{split:<6}{seeded}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    splits = make_splits(load_problems(), load_splits(), args.eval_fraction, args.seed)
    save_splits(splits)
    print(", ".join(f"{name}: {len(ids)}" for name, ids in splits.items()))
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    problems = load_problems()
    if args.ids:
        wanted = set(args.ids.split(","))
        problems = [p for p in problems if p.id in wanted]
    if args.split:
        members = set(load_splits().get(args.split, []))
        problems = [p for p in problems if p.id in members]
    modes = args.modes.split(",")
    failures = 0
    with Runner() as runner:
        stamp = provenance(runner.call("device"))
        for problem in problems:
            dtype = load_spec(problem.spec).dtype(load_spec(problem.spec).variant(args.variant))
            rows = []
            for mode in modes:
                try:
                    timing = _baseline(runner, problem, mode, args)
                except WorkerError as e:
                    failures += 1
                    print(f"{problem.id} {mode}: FAILED {str(e).splitlines()[0]}", flush=True)
                    continue
                if timing is None:
                    continue
                rows.append(
                    {
                        "problem": problem.id,
                        "variant": args.variant,
                        "mode": mode,
                        "dtype": dtype,
                        "time_us": f"{timing.time_us:.2f}",
                        "tflops": f"{timing.tflops:.3f}" if timing.tflops else "",
                        "gbps": f"{timing.gbps:.1f}" if timing.gbps else "",
                        **stamp,
                    }
                )
                print(f"{problem.id} {mode}: {timing.time_us:.1f} us", flush=True)
            upsert(rows)
    return 1 if failures else 0


def _baseline(runner: Runner, problem: Problem, mode: str, args: argparse.Namespace):
    workload = {"spec": problem.spec, "variant": args.variant}
    if mode == "seed":
        return runner.bench(problem.seed, **workload) if problem.seed.exists() else None
    timeout = args.compile_timeout if mode != "eager" else None
    return runner.baseline(problem.reference, mode, timeout=timeout, **workload)


def _probe(runner: Runner, name: str) -> ProbeResult:
    try:
        return ProbeResult.model_validate(runner.call("probe", name=name))
    except WorkerError as e:
        return ProbeResult(name=name, compiled=False, error=str(e))


if __name__ == "__main__":
    sys.exit(main())
