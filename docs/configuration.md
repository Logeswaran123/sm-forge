# Configuration

Settings come from `.env` (copied from `.env.example`) or the environment. Command-line options
override them where both exist.

## LLM backend

| Backend | Setup | Use when |
|---|---|---|
| `api` (default) | `ANTHROPIC_API_KEY=...` in `.env` | you have an Anthropic API key |
| `claude-code` | Claude Code installed and logged in, then `LLM_BACKEND=claude-code` or `--backend claude-code` | you use Claude Code's login instead of a key |

`claude-code` runs `claude -p` headless. It uses `claude` on `PATH`, `CLAUDE_BIN`, or
`--claude-bin`. Opus 5.5 needs Claude Code 2.1.280 or newer. The [Claude Code
engine](claude-code-engine.md) always uses Claude Code.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `api` | `api` or `claude-code` |
| `ANTHROPIC_API_KEY` | | for the `api` backend |
| `CLAUDE_BIN` | `claude` on `PATH` | Claude Code binary |
| `LLM_MODEL` | `claude-opus-5-5` | model |
| `LLM_EFFORT` | `high` | default effort (the planner uses `low`). `medium` costs less, and the published results used it |
| `LLM_MAX_TOKENS` | 64000 | output token cap per call |
| `LLM_BUDGET_USD` | none | stop making LLM calls once a run has spent this much |
| `DEVICE_INDEX` | 0 | GPU the worker runs on |
| `WORKER_TIMEOUT` | 600 | seconds before a worker request is abandoned and the worker restarted |
| `CACHE_DIR` | `.smforge` | runs, builds, reference outputs, logs, workspaces |
| `HARDWARE_DIR` | `hardware` | device profiles |
| `BENCH_WARMUP`, `BENCH_ITERS`, `BENCH_ROUNDS` | 100, 200, 10 | timing loop |
| `BENCH_FLUSH_MB` | 256 | L2 flush between timed calls |
| `BENCH_QUANTILE` | 0.2 | a run's time is this quantile of its samples |
| `BENCHMARK_MIN_EFFECT` | 0.02 | speedups inside this band count as parity |
| `CORRECTNESS_RTOL`, `CORRECTNESS_ATOL` | per dtype | tolerances when the spec gives none |
| `CORRECTNESS_DRAWS` | 2 | input draws per correctness check |
| `CUTLASS_DIR` | fetched into `.smforge/cutlass/` | an existing CUTLASS checkout |

## What lands where

| Path | Contents |
|---|---|
| `hardware/<gpu>.yaml` | device profile from `smforge device --all` |
| `results/*.csv` | one table per command, `*_cuda.csv` for `--dsl cuda` |
| `.smforge/runs/<run>/` | every candidate, its outcome, and each stage's output |
| `.smforge/workspaces/<run>/` | Claude Code workspaces |
| `.smforge/lessons/<gpu>/` | lessons ledgers |
| `.smforge/cuda/`, `.smforge/cutlass/` | CUDA build cache, CUTLASS headers |
