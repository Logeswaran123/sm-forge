"""smforge: optimize Triton and CUDA C++ kernels with an LLM, checked and timed on your GPU."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from smforge import render
from smforge.agents.candidate import spread
from smforge.agents.optimize import optimize_stage
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
    TILESEARCH,
    TILESEARCH_FIELDS,
    TILESEARCH_KEY,
    provenance,
    upsert,
)
from smforge.device.profile import find_profile, load_profile, profile_path, save_profile
from smforge.device.tools import check_tools, tool_status
from smforge.harness.ncu import NcuUnavailable
from smforge.harness.runner import Runner, WorkerError
from smforge.harness.sanitizer import SanitizerUnavailable
from smforge.kb.loader import TOKEN_BUDGET, KnowledgeBase, kb_dir, load_kb, record_measurement
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
from smforge.llm.prompting import PROMPTS
from smforge.models import (
    DeviceInfo,
    DeviceProfile,
    Dsl,
    NoiseResult,
    ProbeResult,
    Roofline,
    Stage,
)
from smforge.spec import load_spec

STAGE_PROMPTS = {path.stem for path in (PROMPTS / "stages").glob("*.md")}
DSLS = [dsl.value for dsl in Dsl]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smforge", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    device = commands.add_parser(
        "device", help="query the GPU and measure its bandwidth, TFLOPS and timing noise"
    )
    device.add_argument("--probes", action="store_true", help="compile feature probes")
    device.add_argument("--roofline", action="store_true", help="measure bandwidth and TFLOPS")
    device.add_argument("--noise", action="store_true", help="A/A timing noise")
    device.add_argument("--tools", action="store_true", help="check ncu, nsys, compute-sanitizer")
    device.add_argument("--all", action="store_true", help="everything above, then --save")
    device.add_argument("--save", action="store_true", help="write hardware/<gpu>.yaml")
    device.add_argument("--json", action="store_true", help="print JSON")
    device.set_defaults(func=cmd_device)

    corpus = commands.add_parser("corpus", help="list, import, split and time the problems")
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

    seed = commands.add_parser(
        "seed", help="write a starting (seed) kernel for problems with the LLM"
    )
    seed.add_argument("--ids", help="comma-separated problem ids")
    seed.add_argument("--dsl", choices=DSLS, default="triton", help="kernel language")
    seed.add_argument("--source", help="only problems from this source (kernelbench, inference)")
    seed.add_argument("--split", choices=("dev", "eval"))
    seed.add_argument("--max-iters", type=int, default=4, help="attempts per problem")
    seed.add_argument("--budget-usd", type=float, help="stop once the run has spent this much")
    seed.add_argument("--force", action="store_true", help="replace existing seeds")
    seed.add_argument("--backend", choices=("api", "claude-code"), help="LLM backend")
    seed.add_argument("--claude-bin", help="Claude Code binary for --backend claude-code")
    seed.set_defaults(func=cmd_seed)

    optimize = commands.add_parser(
        "optimize", help="run one optimization stage on problems with the LLM"
    )
    optimize.add_argument("--stage", required=True, choices=sorted(STAGE_PROMPTS))
    optimize.add_argument("--ids", help="comma-separated problem ids")
    optimize.add_argument("--dsl", choices=DSLS, default="triton", help="kernel language")
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
        "pipeline", help="analyze, plan and run the optimization stages on problems"
    )
    pipeline.add_argument("--ids", help="comma-separated problem ids")
    pipeline.add_argument("--dsl", choices=DSLS, default="triton", help="kernel language")
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
        "claude", help="optimize problems in a headless Claude Code session"
    )
    claude.add_argument("--ids", help="comma-separated problem ids")
    claude.add_argument("--dsl", choices=DSLS, default="triton", help="kernel language")
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

    tiles = commands.add_parser("tile-search", help="search CUTLASS GEMM tile configs with the LLM")
    tiles.add_argument(
        "--shapes", required=True, help="comma-separated MxNxK, optionally :bfloat16 and :row"
    )
    tiles.add_argument("--rounds", type=int, default=3, help="proposal rounds per shape")
    tiles.add_argument("--per-round", type=int, default=6, help="configs per round")
    tiles.add_argument("--effort", help="proposer effort (low, medium, high, xhigh, max)")
    tiles.add_argument("--budget-usd", type=float, help="stop once the run has spent this much")
    tiles.add_argument("--backend", choices=("api", "claude-code"), help="LLM backend")
    tiles.add_argument("--claude-bin", help="Claude Code binary for --backend claude-code")
    tiles.set_defaults(func=cmd_tile_search)

    evaluation = commands.add_parser(
        "eval", help="run the pipeline over a split with one fixed config, and report"
    )
    eval_commands = evaluation.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_commands.add_parser("run", help="run and measure; resumes where it stopped")
    eval_run.add_argument("--name", required=True, help="the run's directory under results/eval/")
    eval_run.add_argument("--split", choices=("dev", "eval"), default="eval")
    eval_run.add_argument("--ids", help="comma-separated problem ids (instead of the whole split)")
    eval_run.add_argument("--dsl", choices=DSLS, default="triton", help="kernel language")
    eval_run.add_argument("--max-iters", type=int, default=2, help="candidates per stage (T)")
    eval_run.add_argument("--max-stages", type=int, default=2, help="stages besides autotuning")
    eval_run.add_argument("--best-of", type=int, default=1, help="pipeline runs, fastest kept")
    eval_run.add_argument("--model", help="LLM model")
    eval_run.add_argument("--effort", default="medium", help="optimizer effort")
    eval_run.add_argument("--keep", type=int, default=2, help="earlier attempts kept in context")
    for flag, text in (
        ("--no-kb", "no knowledge base in the prompts"),
        ("--no-arch-gating", "knowledge base entries for every GPU"),
        ("--no-facts", "prompts show the timing only"),
        ("--no-reanalysis", "no re-measurement after an accepted stage"),
        ("--no-rollback", "keep slower stage outputs"),
        ("--no-ncu", "no Nsight Compute counters"),
        ("--no-memcheck", "finalize without compute-sanitizer memcheck"),
    ):
        eval_run.add_argument(flag, action="store_true", help=text)
    eval_run.add_argument("--budget-usd", type=float, help="stop once the run has spent this much")
    eval_run.add_argument("--backend", choices=("api", "claude-code"), help="LLM backend")
    eval_run.add_argument("--claude-bin", help="Claude Code binary for --backend claude-code")
    eval_run.set_defaults(func=cmd_eval_run)
    eval_report = eval_commands.add_parser("report", help="Markdown tables for one or more runs")
    eval_report.add_argument(
        "--runs", required=True, help="comma-separated run names; the first is the baseline"
    )
    eval_report.add_argument("--output", type=Path, help="write here instead of printing")
    eval_report.set_defaults(func=cmd_eval_report)

    kb = commands.add_parser("kb", help="the knowledge base of constraints and patterns")
    kb_commands = kb.add_subparsers(dest="kb_command", required=True)

    kb_list = kb_commands.add_parser("list", help="entries with their stage and measured speedup")
    kb_list.add_argument("--dsl", choices=DSLS, default="triton", help="kernel language")
    kb_list.set_defaults(func=cmd_kb_list)

    show = kb_commands.add_parser("show", help="a stage's slice as a prompt receives it")
    show.add_argument("stage", choices=[stage.value for stage in Stage])
    show.add_argument("--families", help="comma-separated corpus families")
    show.add_argument("--budget", type=int, default=TOKEN_BUDGET, help="token budget")
    show.add_argument("--dsl", choices=DSLS, default="triton", help="kernel language")
    show.set_defaults(func=cmd_kb_show)

    verify = kb_commands.add_parser("verify", help="run every entry's kernel pair on the GPU")
    verify.add_argument("--ids", help="comma-separated entry ids")
    verify.add_argument(
        "--write", action="store_true", help="record example speedups in measured.yaml"
    )
    verify.add_argument("--dsl", choices=DSLS, default="triton", help="kernel language")
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
    kb = load_kb(kb_dir(_dsl(args)))
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
            upsert([row], _results(STAGES, args), STAGE_FIELDS, STAGE_KEY)
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
    kb = load_kb(kb_dir(_dsl(args)))
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
                    _ledger(config.cache_dir, stamp["gpu"], problem),
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
                        "ab": "/".join(f"{x:.3f}" for x in result.ab),
                        "ab_spread": f"{spread(result.ab):.3f}" if result.ab else "",
                        "model": config.llm.model,
                        **stamp,
                    }
                ],
                _results(PIPELINE, args),
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
                _results(PIPELINE_STAGES, args),
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
    kb = load_kb(kb_dir(_dsl(args)))
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
                    _ledger(config.cache_dir, stamp["gpu"], problem),
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
            if result.ab:
                runs = "/".join(f"{x:.3f}" for x in result.ab)
                line += f"; trial {result.trial_speedup:.3f}x, re-measured {runs}x"
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
                        "ab": "/".join(f"{x:.3f}" for x in result.ab),
                        "ab_spread": f"{spread(result.ab):.3f}" if result.ab else "",
                        "turns": session.turns,
                        "stop": session.stop,
                        "cost_usd": f"{session.cost_usd:.4f}",
                        "model": config.llm.model,
                        **stamp,
                    }
                ],
                _results(CLAUDE, args),
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


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return format(value, spec) if value is not None else ""


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
    """The selected problems, for ``--dsl``'s seed."""
    problems = [p.for_dsl(_dsl(args)) for p in problems]
    if getattr(args, "ids", None):
        wanted = set(args.ids.split(","))
        problems = [p for p in problems if p.id in wanted]
    if getattr(args, "source", None):
        problems = [p for p in problems if p.source == args.source]
    if getattr(args, "split", None):
        members = set(load_splits().get(args.split, []))
        problems = [p for p in problems if p.id in members]
    return problems


def _dsl(args: argparse.Namespace) -> Dsl:
    return Dsl(getattr(args, "dsl", None) or Dsl.TRITON)


def _results(path: Path, args: argparse.Namespace) -> Path:
    """``results/x.csv`` for Triton, ``results/x_cuda.csv`` for CUDA."""
    dsl = _dsl(args)
    return path if dsl is Dsl.TRITON else path.with_name(f"{path.stem}_{dsl}{path.suffix}")


def _ledger(cache_dir: Path, gpu: str, problem: Problem) -> Path:
    name = problem.id if problem.dsl is Dsl.TRITON else f"{problem.id}.{problem.dsl}"
    return ledger_path(cache_dir, gpu, name)


def _probe(runner: Runner, name: str) -> ProbeResult:
    try:
        return ProbeResult.model_validate(runner.call("probe", name=name))
    except WorkerError as e:
        return ProbeResult(name=name, compiled=False, error=str(e))


def cmd_tile_search(args: argparse.Namespace) -> int:
    from smforge.tilesearch import GemmShape, search, unsupported

    config = _llm_config(args)
    if args.effort:
        config.llm.efforts["tilesearch"] = args.effort
    shapes = [GemmShape.parse(text) for text in args.shapes.split(",")]
    for shape in shapes:
        if reason := unsupported(shape):
            print(f"{shape.label}: not searched: {reason}", file=sys.stderr)
    shapes = [shape for shape in shapes if not unsupported(shape)]
    profile = find_profile(config.hardware_dir, config.device)
    if profile is None:
        print("no hardware profile: run `smforge device` first", file=sys.stderr)
        return 2
    run = time.strftime("%Y%m%d-%H%M%S")
    llm = create_client(config.llm, transcripts=config.cache_dir / "transcripts" / run)
    with Runner(config) as runner:
        stamp = provenance(runner.call("device"))
        for shape in shapes:
            try:
                result = search(
                    shape,
                    runner,
                    llm,
                    profile,
                    config.cache_dir / "runs" / run / "tilesearch",
                    args.rounds,
                    args.per_round,
                )
            except BudgetExceeded as e:
                print(f"{e}\nstopped at {shape.label}", file=sys.stderr)
                return 2
            best, fitted = result.best, result.fitted
            print(
                f"{shape.label}: cuBLAS {result.cublas_us:.1f} us; default "
                f"{result.default.row}; fitted {fitted.row}",
                flush=True,
            )
            if best is not None:
                print(
                    f"  best {best.config.label}: A/B {_fmt(result.best_us, '.1f')} us "
                    f"({result.tflops(result.best_us):.1f} TFLOPS) against default "
                    f"{_fmt(result.default_us, '.1f')} us ({_fmt(result.vs_default)}x) and "
                    f"cuBLAS {_fmt(result.final_cublas_us, '.1f')} us ({_fmt(result.vs_cublas)}x)",
                    flush=True,
                )
            upsert(
                [
                    {
                        "shape": shape.label,
                        "cublas_us": _fmt(result.final_cublas_us or result.cublas_us, ".2f"),
                        "default": fitted.config.label
                        + ("" if fitted.us else f" ({fitted.status})"),
                        "default_us": _fmt(result.default_us or fitted.us, ".2f"),
                        "best": best.config.label if best else "",
                        "best_us": _fmt(result.best_us, ".2f"),
                        "best_tflops": _fmt(result.tflops(result.best_us), ".1f"),
                        "vs_default": _fmt(result.vs_default),
                        "vs_cublas": _fmt(result.vs_cublas),
                        "configs": len(result.trials),
                        "invalid": sum(t.status == "invalid" for t in result.trials),
                        "failed": sum(
                            t.us is None and t.status != "invalid" for t in result.trials
                        ),
                        "cost_usd": f"{result.usage.cost_usd:.4f}",
                        "model": config.llm.model,
                        **stamp,
                    }
                ],
                TILESEARCH,
                TILESEARCH_FIELDS,
                TILESEARCH_KEY,
            )
    return 0


EVAL_DIR = Path("results") / "eval"


def cmd_eval_run(args: argparse.Namespace) -> int:
    from smforge.evaluation import RunConfig, evaluate, save, summarize

    if args.ids:
        args.split = None
    config = _llm_config(args)
    if args.model:
        config.llm.model = args.model
    config.llm.efforts["optimizer"] = args.effort
    run = RunConfig(
        split=None if args.ids else args.split,
        dsl=args.dsl,
        model=config.llm.model,
        effort=args.effort,
        max_iters=args.max_iters,
        max_stages=args.max_stages,
        best_of=args.best_of,
        kb=not args.no_kb,
        arch_gating=not args.no_arch_gating,
        facts=not args.no_facts,
        reanalysis=not args.no_reanalysis,
        rollback=not args.no_rollback,
        ncu=not args.no_ncu and _tool_ok("ncu", "running without counters"),
        memcheck=not args.no_memcheck and _tool_ok("compute-sanitizer", "finals unchecked"),
    )
    run_dir = EVAL_DIR / args.name
    config_file = run_dir / "config.json"
    if config_file.exists():
        if RunConfig.model_validate_json(config_file.read_text()) != run:
            print(f"{run_dir} was run with another config; use another --name", file=sys.stderr)
            return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(run.model_dump_json(indent=1))
    problems = [p for p in _select(load_problems(), args) if p.seed.exists()]
    todo = [p for p in problems if not (run_dir / f"{p.id}.json").exists()]
    print(f"{args.name}: {len(problems) - len(todo)} of {len(problems)} done", flush=True)
    profile = find_profile(config.hardware_dir, config.device)
    kb = load_kb(kb_dir(_dsl(args))) if run.kb else KnowledgeBase(kb_dir(_dsl(args)))
    if profile is not None and run.arch_gating:
        kb = kb.for_device(profile)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    llm = create_client(config.llm, transcripts=config.cache_dir / "transcripts" / stamp)
    with Runner(config) as runner:
        # the first session's provenance stays: a resumed run keeps the hash it started on
        if not (run_dir / "device.json").exists():
            (run_dir / "device.json").write_text(
                json.dumps(provenance(runner.call("device")), indent=1)
            )
        for problem in todo:
            workdir = config.cache_dir / "runs" / stamp / args.name / problem.id
            try:
                result = evaluate(problem, runner, llm, kb, profile, run, workdir, args.keep)
            except BudgetExceeded as e:
                print(f"stopping: {e}; rerun the same command to resume", flush=True)
                break
            except (MissingCredentials, ClaudeCodeError, NcuUnavailable, SanitizerUnavailable) as e:
                print(f"{e}\nstopped at {problem.id}; rerun to resume", file=sys.stderr)
                return 2
            except (Refused, ReplayMiss, WorkerError) as e:
                print(f"{problem.id}: FAILED: {str(e).splitlines()[0]}", flush=True)
                continue
            save(result, run_dir)
            held = [f"{k} {v}" for k, v in result.correctness.items() if v != "pass"]
            print(
                f"{problem.id}: {result.speedup:.2f}x over the seed"
                + (f", {s:.2f}x over eager" if (s := result.over("eager")) else "")
                + f", held-out {', '.join(held) or 'pass'}, ${result.llm.cost_usd:.2f}, "
                f"{result.wall_s / 60:.1f} min",
                flush=True,
            )
    from smforge.evaluation import load_run

    _, done = load_run(run_dir)
    summary = summarize(done, config.timing.min_effect)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(
        f"{args.name}: {summary['problems']} problems, {summary['improved']} improved, "
        f"{summary['regressions']} regressions, geomean {summary['geomean_over_seed'] or 0:.2f}x "
        "over the seed",
        flush=True,
    )
    return 0


def cmd_eval_report(args: argparse.Namespace) -> int:
    from smforge.evaluation import report

    names = args.runs.split(",")
    missing = [n for n in names if not (EVAL_DIR / n / "config.json").exists()]
    if missing:
        print(f"no run: {', '.join(missing)}", file=sys.stderr)
        return 2
    text = report({name: EVAL_DIR / name for name in names})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


def cmd_kb_list(args: argparse.Namespace) -> int:
    kb = load_kb(kb_dir(_dsl(args)))
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
    kb = load_kb(kb_dir(_dsl(args)))
    profile = find_profile(config.hardware_dir, config.device)
    if profile is not None:
        kb = kb.for_device(profile)
    families = args.families.split(",") if args.families else ()
    print(kb.render(Stage(args.stage), families, args.budget))
    return 0


def cmd_kb_verify(args: argparse.Namespace) -> int:
    kb = load_kb(kb_dir(_dsl(args)))
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
