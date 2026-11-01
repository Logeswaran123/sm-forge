"""smforge: device baseline and the problem corpus; the optimization pipeline in later stages."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from smforge import render
from smforge.agents.optimize import PROMPTS, optimize_stage
from smforge.agents.pipeline import memcheck_summary, run_pipeline
from smforge.agents.seed import seed_problem
from smforge.claude_ws import ClaudeSessionError, run_claude
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
from smforge.corpus.results import (
    CLAUDE,
    CLAUDE_FIELDS,
    PIPELINE,
    PIPELINE_FIELDS,
    PIPELINE_KEY,
    PIPELINE_STAGES,
    STAGE_FIELDS,
    STAGE_KEY,
    STAGES,
    provenance,
    upsert,
)
from smforge.device.profile import find_profile, load_profile, profile_path, save_profile
from smforge.device.tools import check_tools, tool_status
from smforge.harness.ncu import NcuUnavailable
from smforge.harness.runner import Runner, WorkerError
from smforge.harness.sanitizer import SanitizerUnavailable
from smforge.kb.loader import TOKEN_BUDGET, load_kb, record_measurement
from smforge.kb.verify import Outcome, verify_constraint, verify_example
from smforge.lessons import ledger_path
from smforge.llm.claude_code import ClaudeCodeError
from smforge.llm.client import (
    BudgetExceeded,
    MissingCredentials,
    Refused,
    ReplayMiss,
    create_client,
)
from smforge.models import DeviceInfo, DeviceProfile, NoiseResult, ProbeResult, Roofline, Stage
from smforge.spec import load_spec

STAGE_PROMPTS = {path.stem for path in (PROMPTS / "stages").glob("*.md")}


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
    baselines.add_argument("--source", help="only problems from this source")
    baselines.add_argument("--split", choices=("dev", "eval"))
    baselines.add_argument("--modes", default="eager,compile,seed")
    baselines.add_argument("--variant", default="bench-gpu")
    baselines.add_argument(
        "--compile-timeout", type=float, default=1800, help="seconds allowed per compile"
    )
    baselines.set_defaults(func=cmd_baselines)

    seed = commands.add_parser("seed", help="write a seed kernel for corpus problems with the LLM")
    seed.add_argument("--ids", help="comma-separated problem ids")
    seed.add_argument("--source", help="only problems from this source (kernelbench, inference)")
    seed.add_argument("--split", choices=("dev", "eval"))
    seed.add_argument("--max-iters", type=int, default=4, help="attempts per problem")
    seed.add_argument("--budget-usd", type=float, help="stop once the run has spent this much")
    seed.add_argument("--force", action="store_true", help="replace existing seeds")
    seed.add_argument("--backend", choices=("api", "claude-code"), help="LLM backend")
    seed.add_argument("--claude-bin", help="Claude Code binary for --backend claude-code")
    seed.set_defaults(func=cmd_seed)

    optimize = commands.add_parser(
        "optimize", help="run an optimization stage on corpus problems with the LLM (stage 4)"
    )
    optimize.add_argument("--stage", required=True, choices=sorted(STAGE_PROMPTS))
    optimize.add_argument("--ids", help="comma-separated problem ids")
    optimize.add_argument("--source", help="only problems from this source")
    optimize.add_argument("--split", choices=("dev", "eval"))
    optimize.add_argument("--max-iters", type=int, default=3, help="candidates per problem")
    optimize.add_argument("--keep", type=int, default=2, help="earlier attempts kept in context")
    optimize.add_argument("--budget-usd", type=float, help="stop once the run has spent this much")
    optimize.add_argument("--backend", choices=("api", "claude-code"), help="LLM backend")
    optimize.add_argument("--claude-bin", help="Claude Code binary for --backend claude-code")
    optimize.add_argument("--record", type=Path, help="append every reply to this JSONL file")
    optimize.add_argument("--replay", type=Path, help="answer from a recording, without the LLM")
    optimize.set_defaults(func=cmd_optimize)

    pipeline = commands.add_parser(
        "pipeline", help="analyze, plan and run every stage on corpus problems (stage 5)"
    )
    pipeline.add_argument("--ids", help="comma-separated problem ids")
    pipeline.add_argument("--source", help="only problems from this source")
    pipeline.add_argument("--split", choices=("dev", "eval"))
    pipeline.add_argument("--max-iters", type=int, default=2, help="candidates per stage")
    pipeline.add_argument(
        "--max-stages", type=int, default=3, help="stages per problem besides autotuning"
    )
    pipeline.add_argument("--effort", help="optimizer effort (low, medium, high, xhigh, max)")
    pipeline.add_argument("--keep", type=int, default=2, help="earlier attempts kept in context")
    pipeline.add_argument("--budget-usd", type=float, help="stop once the run has spent this much")
    pipeline.add_argument("--backend", choices=("api", "claude-code"), help="LLM backend")
    pipeline.add_argument("--claude-bin", help="Claude Code binary for --backend claude-code")
    pipeline.add_argument("--record", type=Path, help="append every reply to this JSONL file")
    pipeline.add_argument("--replay", type=Path, help="answer from a recording, without the LLM")
    pipeline.add_argument(
        "--from", dest="start", metavar="ID", help="skip the selected problems before this one"
    )
    pipeline.add_argument("--no-ncu", action="store_true", help="run without Nsight Compute")
    pipeline.add_argument(
        "--no-memcheck", action="store_true", help="finalize without compute-sanitizer memcheck"
    )
    pipeline.set_defaults(func=cmd_pipeline)

    claude = commands.add_parser(
        "claude", help="optimize corpus problems in a headless Claude Code workspace (stage 6)"
    )
    claude.add_argument("--ids", help="comma-separated problem ids")
    claude.add_argument("--source", help="only problems from this source")
    claude.add_argument("--split", choices=("dev", "eval"))
    claude.add_argument(
        "--max-trials", type=int, default=8, help="trials the session is told to try"
    )
    claude.add_argument("--max-turns", type=int, default=80, help="turn cap for the session")
    claude.add_argument("--budget-usd", type=float, help="dollar cap per session")
    claude.add_argument("--effort", help="session effort (low, medium, high, xhigh, max)")
    claude.add_argument("--claude-bin", help="Claude Code binary")
    claude.add_argument("--no-ncu", action="store_true", help="measure t0 without Nsight Compute")
    claude.add_argument(
        "--no-memcheck", action="store_true", help="finalize without compute-sanitizer memcheck"
    )
    claude.set_defaults(func=cmd_claude)

    kb = commands.add_parser("kb", help="the knowledge base of constraints and patterns (stage 3)")
    kb_commands = kb.add_subparsers(dest="kb_command", required=True)

    kb_list = kb_commands.add_parser("list", help="entries with their stage and measured speedup")
    kb_list.set_defaults(func=cmd_kb_list)

    show = kb_commands.add_parser("show", help="a stage's slice as a prompt receives it")
    show.add_argument("stage", choices=[stage.value for stage in Stage])
    show.add_argument("--families", help="comma-separated corpus families")
    show.add_argument("--budget", type=int, default=TOKEN_BUDGET, help="token budget")
    show.set_defaults(func=cmd_kb_show)

    verify = kb_commands.add_parser("verify", help="run every entry's kernel pair on the GPU")
    verify.add_argument("--ids", help="comma-separated entry ids")
    verify.add_argument(
        "--write", action="store_true", help="record example speedups in measured.yaml"
    )
    verify.set_defaults(func=cmd_kb_verify)
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
    problems = _select(load_problems(), args)
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


def cmd_seed(args: argparse.Namespace) -> int:
    config = _llm_config(args)
    problems = _select(load_problems(), args)
    if not args.force:
        problems = [p for p in problems if not p.seed.exists()]
    run = time.strftime("%Y%m%d-%H%M%S")
    llm = create_client(config.llm, transcripts=config.cache_dir / "transcripts" / run)
    profile = find_profile(config.hardware_dir, config.device)
    seeded = 0
    with Runner(config) as runner:
        for problem in problems:
            workdir = config.cache_dir / "seeds" / run / problem.id
            try:
                result = seed_problem(problem, runner, llm, profile, args.max_iters, workdir)
            except BudgetExceeded as e:
                print(f"stopping: {e}", flush=True)
                break
            except (MissingCredentials, ClaudeCodeError) as e:
                print(e, file=sys.stderr)
                return 2
            except Refused as e:
                print(f"{problem.id}: REFUSED {e}", flush=True)
                continue
            cost = f"${result.usage.cost_usd:.2f}"
            if result.passed:
                seeded += 1
                print(f"{problem.id}: seeded in {len(result.attempts)} ({cost})", flush=True)
            else:
                last = result.attempts[-1].feedback.splitlines()[0] if result.attempts else ""
                print(f"{problem.id}: FAILED after {len(result.attempts)} ({cost}): {last}")
    print(
        f"seeded {seeded} of {len(problems)}; spent ${llm.usage.cost_usd:.2f} in {llm.usage.calls} calls"
    )
    return 0 if seeded == len(problems) else 1


def cmd_optimize(args: argparse.Namespace) -> int:
    config = _llm_config(args)
    stage = Stage(args.stage)
    problems = [p for p in _select(load_problems(), args) if p.seed.exists()]
    run = time.strftime("%Y%m%d-%H%M%S")
    llm = create_client(
        config.llm,
        transcripts=config.cache_dir / "transcripts" / run,
        record=args.record,
        replay=args.replay,
    )
    profile = find_profile(config.hardware_dir, config.device)
    kb = load_kb()
    if profile is not None:
        kb = kb.for_device(profile)
    accepted = 0
    with Runner(config) as runner:
        stamp = provenance(runner.call("device"))
        for problem in problems:
            workdir = config.cache_dir / "runs" / run / problem.id / stage.value
            try:
                result = optimize_stage(
                    problem,
                    stage,
                    problem.seed,
                    runner,
                    llm,
                    kb,
                    profile,
                    args.max_iters,
                    workdir,
                    args.keep,
                )
            except BudgetExceeded as e:
                print(f"stopping: {e}", flush=True)
                break
            except (MissingCredentials, ClaudeCodeError) as e:
                print(e, file=sys.stderr)
                return 2
            except (Refused, ReplayMiss) as e:
                print(f"{problem.id}: {type(e).__name__} {e}", flush=True)
                continue
            except WorkerError as e:
                print(f"{problem.id}: FAILED to measure the input: {str(e).splitlines()[0]}")
                continue
            accepted += result.accepted
            outcomes = ",".join(a.outcome for a in result.attempts)
            verdict = f"{result.speedup:.2f}x" if result.accepted else "kept the input"
            cost = f"${result.usage.cost_usd:.2f}"
            print(f"{problem.id}: {verdict} [{outcomes}] ({cost})", flush=True)
            row = {
                "problem": problem.id,
                "stage": stage.value,
                "accepted": int(result.accepted),
                "input_us": f"{result.input_us:.2f}",
                "output_us": f"{result.output_us:.2f}",
                "speedup": f"{result.speedup:.3f}",
                "attempts": len(result.attempts),
                "outcomes": outcomes,
                "cost_usd": f"{result.usage.cost_usd:.4f}",
                "model": config.llm.model,
                **stamp,
            }
            upsert([row], STAGES, STAGE_FIELDS, STAGE_KEY)
    print(
        f"accepted {accepted} of {len(problems)}; "
        f"spent ${llm.usage.cost_usd:.2f} in {llm.usage.calls} calls"
    )
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    config = _llm_config(args)
    if args.effort:
        config.llm.efforts["optimizer"] = args.effort
    problems = [p for p in _select(load_problems(), args) if p.seed.exists()]
    if args.start:
        ids = [p.id for p in problems]
        if args.start not in ids:
            print(f"--from {args.start}: not among the selected problems", file=sys.stderr)
            return 2
        problems = problems[ids.index(args.start) :]
    ncu = not args.no_ncu and _tool_ok(
        "ncu", "running without counters, so uncoalesced_access and bank_conflict go undetected"
    )
    memcheck = not args.no_memcheck and _tool_ok(
        "compute-sanitizer", "final kernels are not memchecked"
    )
    run = time.strftime("%Y%m%d-%H%M%S")
    llm = create_client(
        config.llm,
        transcripts=config.cache_dir / "transcripts" / run,
        record=args.record,
        replay=args.replay,
    )
    profile = find_profile(config.hardware_dir, config.device)
    kb = load_kb()
    if profile is not None:
        kb = kb.for_device(profile)
    improved = 0
    with Runner(config) as runner:
        stamp = provenance(runner.call("device"))
        for problem in problems:
            workdir = config.cache_dir / "runs" / run / problem.id / "pipeline"
            try:
                result = run_pipeline(
                    problem,
                    runner,
                    llm,
                    kb,
                    profile,
                    args.max_iters,
                    workdir,
                    args.keep,
                    args.max_stages,
                    ncu,
                    memcheck,
                    ledger_path(config.cache_dir, stamp["gpu"], problem.id),
                )
            except BudgetExceeded as e:
                print(f"stopping: {e}; resume with --from {problem.id}", flush=True)
                break
            except (MissingCredentials, ClaudeCodeError, NcuUnavailable, SanitizerUnavailable) as e:
                print(
                    f"{e}\nstopped at {problem.id}; resume with --from {problem.id}",
                    file=sys.stderr,
                )
                return 2
            except (Refused, ReplayMiss) as e:
                print(f"{problem.id}: {type(e).__name__} {e}", flush=True)
                continue
            except WorkerError as e:
                print(f"{problem.id}: FAILED: {str(e).splitlines()[0]}", flush=True)
                continue
            improved += result.kernel != problem.seed
            accepted = [str(s.stage) for s in result.stages if s.accepted]
            stages = (
                " ".join(f"{s.stage}{'+' if s.accepted else '-'}" for s in result.stages)
                or "no stages"
            )
            print(
                f"{problem.id}: {result.speedup:.2f}x [{stages}] (${result.usage.cost_usd:.2f})",
                flush=True,
            )
            upsert(
                [
                    {
                        "problem": problem.id,
                        "seed_us": f"{result.seed_us:.2f}",
                        "final_us": f"{result.final_us:.2f}",
                        "speedup": f"{result.speedup:.3f}",
                        "plan": ",".join(map(str, result.plan)),
                        "accepted": ",".join(accepted),
                        "issues": ",".join(sorted({str(i.type) for i in result.issues})),
                        "cost_usd": f"{result.usage.cost_usd:.4f}",
                        "memcheck": memcheck_summary(result),
                        "model": config.llm.model,
                        **stamp,
                    }
                ],
                PIPELINE,
                PIPELINE_FIELDS,
                PIPELINE_KEY,
            )
            upsert(
                [
                    {
                        "problem": problem.id,
                        "stage": str(s.stage),
                        "accepted": int(s.accepted),
                        "input_us": f"{s.input_us:.2f}",
                        "output_us": f"{s.output_us:.2f}",
                        "speedup": f"{s.speedup:.3f}",
                        "attempts": len(s.attempts),
                        "outcomes": ",".join(a.outcome for a in s.attempts),
                        "cost_usd": f"{s.usage.cost_usd:.4f}",
                        "model": config.llm.model,
                        **stamp,
                    }
                    for s in result.stages
                ],
                PIPELINE_STAGES,
                STAGE_FIELDS,
                STAGE_KEY,
                drop={"problem": problem.id},
            )
    print(
        f"improved {improved} of {len(problems)}; "
        f"spent ${llm.usage.cost_usd:.2f} in {llm.usage.calls} calls"
    )
    return 0


def cmd_claude(args: argparse.Namespace) -> int:
    config = get_config()
    if args.claude_bin:
        config.llm.claude_bin = args.claude_bin
    if args.effort:
        config.llm.efforts["optimizer"] = args.effort
    problems = [p for p in _select(load_problems(), args) if p.seed.exists()]
    ncu = not args.no_ncu and _tool_ok("ncu", "t0 is measured without counters")
    memcheck = not args.no_memcheck and _tool_ok(
        "compute-sanitizer", "final kernels are not memchecked"
    )
    run = time.strftime("%Y%m%d-%H%M%S")
    profile = find_profile(config.hardware_dir, config.device)
    kb = load_kb()
    if profile is not None:
        kb = kb.for_device(profile)
    spent = 0.0
    with Runner(config) as runner:
        stamp = provenance(runner.call("device"))
        for problem in problems:
            workdir = config.cache_dir / "workspaces" / run / problem.id
            try:
                result = run_claude(
                    problem,
                    runner,
                    kb,
                    profile,
                    config,
                    workdir,
                    args.max_trials,
                    args.max_turns,
                    args.budget_usd,
                    ledger_path(config.cache_dir, stamp["gpu"], problem.id),
                    ncu,
                    memcheck,
                )
            except (ClaudeSessionError, NcuUnavailable) as e:
                print(f"{e}\nstopped at {problem.id}", file=sys.stderr)
                return 2
            except WorkerError as e:
                print(f"{problem.id}: FAILED: {str(e).splitlines()[0]}", flush=True)
                continue
            spent += result.session.cost_usd
            session = result.session
            line = (
                f"{problem.id}: {result.speedup:.2f}x ({result.final or 'seed kept'}; "
                f"{result.trials} trials, {session.turns} turns, {session.stop}, "
                f"${session.cost_usd:.2f})"
            )
            if result.remeasured is not None:
                line += f"; trial {result.trial_speedup:.3f}x, re-measured {result.remeasured:.3f}x"
            print(line, flush=True)
            for note in result.notes:
                print(f"  {note}", flush=True)
            print(f"  workspace: {result.workspace}", flush=True)
            upsert(
                [
                    {
                        "problem": problem.id,
                        "seed_us": f"{result.seed_us:.2f}",
                        "final_us": f"{result.final_us:.2f}",
                        "speedup": f"{result.speedup:.3f}",
                        "final_trial": result.final or "",
                        "trials": result.trials,
                        "trial_speedup": _fmt(result.trial_speedup),
                        "remeasured": _fmt(result.remeasured),
                        "turns": session.turns,
                        "stop": session.stop,
                        "cost_usd": f"{session.cost_usd:.4f}",
                        "model": config.llm.model,
                        **stamp,
                    }
                ],
                CLAUDE,
                CLAUDE_FIELDS,
                PIPELINE_KEY,
            )
    print(f"spent ${spent:.2f} on {len(problems)} problems")
    return 0


def _tool_ok(name: str, consequence: str) -> bool:
    status = tool_status(name)
    if not status.ok:
        print(f"{name} unavailable ({status.detail}): {consequence}", flush=True)
    return status.ok


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else ""


def _llm_config(args: argparse.Namespace):
    config = get_config()
    if args.budget_usd is not None:
        config.llm.budget_usd = args.budget_usd
    if args.backend:
        config.llm.backend = args.backend
    if args.claude_bin:
        config.llm.claude_bin = args.claude_bin
    return config


def _select(problems: list[Problem], args: argparse.Namespace) -> list[Problem]:
    if getattr(args, "ids", None):
        wanted = set(args.ids.split(","))
        problems = [p for p in problems if p.id in wanted]
    if getattr(args, "source", None):
        problems = [p for p in problems if p.source == args.source]
    if getattr(args, "split", None):
        members = set(load_splits().get(args.split, []))
        problems = [p for p in problems if p.id in members]
    return problems


def _probe(runner: Runner, name: str) -> ProbeResult:
    try:
        return ProbeResult.model_validate(runner.call("probe", name=name))
    except WorkerError as e:
        return ProbeResult(name=name, compiled=False, error=str(e))


def cmd_kb_list(args: argparse.Namespace) -> int:
    kb = load_kb()
    for c in kb.constraints:
        stages = ",".join(c.stages) or "all"
        print(f"constraint  {c.id:44s} {stages:30s} {c.check.problem}")
    for p in kb.patterns:
        example = kb.examples[p.example]
        speedups = ", ".join(f"{m.speedup:.2f}x {m.gpu}" for m in example.measured) or "-"
        print(f"pattern     {p.id:44s} {p.stage:30s} {example.problem}  {speedups}")
    print(f"{len(kb.constraints)} constraints, {len(kb.patterns)} patterns")
    return 0


def cmd_kb_show(args: argparse.Namespace) -> int:
    config = get_config()
    kb = load_kb()
    profile = find_profile(config.hardware_dir, config.device)
    if profile is not None:
        kb = kb.for_device(profile)
    families = args.families.split(",") if args.families else ()
    print(kb.render(Stage(args.stage), families, args.budget))
    return 0


def cmd_kb_verify(args: argparse.Namespace) -> int:
    kb = load_kb()
    wanted = set(args.ids.split(",")) if args.ids else None
    problems = {p.id: p for p in load_problems()}
    dev = set(load_splits().get("dev", []))
    outcomes = []

    def report(outcome: Outcome) -> None:
        outcomes.append(outcome)
        status = "PASS" if outcome.passed else "FAIL"
        print(f"{status} {outcome.kind} {outcome.id}: {outcome.detail}", flush=True)

    with Runner() as runner:
        device = DeviceInfo.model_validate(runner.call("device"))
        for constraint in kb.constraints:
            if wanted is None or constraint.id in wanted:
                report(verify_constraint(kb, constraint, runner, problems, dev))
        examples: dict[str, Outcome] = {}
        for pattern in kb.patterns:
            if wanted is not None and pattern.id not in wanted:
                continue
            example = kb.examples[pattern.example]
            if example.id not in examples:
                outcome = verify_example(kb, example, runner, problems, dev, device)
                if args.write and outcome.passed:
                    record_measurement(kb, example.id, outcome.measurement)
                examples[example.id] = outcome
            report(replace(examples[example.id], id=pattern.id, kind="pattern"))
    passed = sum(o.passed for o in outcomes)
    print(f"{passed} of {len(outcomes)} entries verified")
    return 0 if passed == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
