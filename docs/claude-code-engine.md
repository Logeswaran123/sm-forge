# Claude Code engine

`smforge claude` hands each problem to a headless Claude Code session in a workspace of its own,
instead of the pipeline's fixed stages. The session decides what to try, and sm-forge checks and
times every attempt and picks the winner itself.

```bash
smforge claude --ids l1_035_groupnorm --max-trials 8 --max-turns 80 --budget-usd 10
smforge claude --dsl cuda --ids l1_026_gelu
```

> [!IMPORTANT]
> It needs Claude Code installed and logged in: `claude` on `PATH`, or `CLAUDE_BIN`, or
> `--claude-bin`. Opus 5.5 needs Claude Code 2.1.280 or newer.

## The workspace

`.smforge/workspaces/<run>/<problem>/` holds:

| Path | What it is |
|---|---|
| `kernel.py` | the seed, also trial `t0` |
| `problem/` | read-only copies of the reference and spec |
| `CLAUDE.md` | the rules, the checks a trial goes through, and the measurements of `t0` |
| `kb.md`, `stages.md` | the knowledge base filtered for this GPU, and the stage instructions |
| `lessons.md` | the problem's lessons ledger from earlier runs |
| `work/` | the only place the session may write |
| `trials/` | the trial tree |
| `transcript.jsonl` | the session's transcript |

The session may run only `smforge-skill` (and read-only shell commands). It saves each attempt
with `smforge-skill trial save`, which checks it and times it against `t0` at once
([trial tree](harness.md#trial-tree)).

## Finalizing

When the session stops, or reaches `--max-turns`, sm-forge takes the best trial, checks it again
in its own worker against the corpus reference, runs memcheck, and then three A/B runs against
the seed in fresh workers, the slowest counting. A trial that fails gives way to the next best.

```
l1_035_groupnorm: 1.27x (t3; 5 trials, 30 turns, success, $0.87); trial 1.590x, re-measured 1.579/1.573/1.275x
```

Results go to `results/claude.csv` (`results/claude_cuda.csv` with `--dsl cuda`). A session that
fails before its first trial stops the run.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--max-trials` | 8 | trials the session is told to try |
| `--max-turns` | 80 | turn cap for the session |
| `--budget-usd` | none | dollar cap per session |
| `--effort` | from `.env` | session effort |
| `--no-ncu`, `--no-memcheck` | off | measure `t0` without counters, finalize without memcheck |
