# CUTLASS tile search

`smforge tile-search` tunes CUTLASS's SM80 GEMM for given shapes. Each round the LLM proposes
configs, the harness builds, checks and times them, and the LLM sees every result before it
proposes again. The best config is compared with cuBLAS and with CUTLASS's default config.

```bash
smforge tile-search --shapes 4096x4096x4096:row,1024x8192x8192 --rounds 3 --per-round 6
```

```
1024x8192x8192_float16_col: cuBLAS 1219.6 us; default 128x256x64_w64x64x64_s3: invalid, - (needs 144 KB of shared memory; a block gets 99 KB); fitted 128x256x64_w64x64x64_s2: ok, 1518.6 us
  best 64x64x64_w32x32x64_s3_sw2: A/B 1167.2 us (117.8 TFLOPS) against default 1516.5 us (1.299x) and cuBLAS 1224.7 us (1.048x)
```

## Shapes

A shape is `MxNxK` for `C[M, N] = A[M, K] @ B`, with:

- B stored `[N, K]` as in `nn.Linear`, or `[K, N]` with `:row`
- fp16, or bf16 with `:bfloat16`

K and N must be even: `cp.async` copies at least 4 bytes, so rows of A and B must start on
4-byte boundaries. Other shapes are reported and skipped.

## What is searched

| Parameter | Values |
|---|---|
| threadblock tile M, N, K | typically 64 to 256 for M and N, 32 or 64 for K |
| warp tile M, N, K | divides the threadblock tile, at most 128 accumulators per thread |
| stages | 2 to 8, within the shared memory per block (99 KB on sm_120) |
| swizzle | 1, 2, 4 or 8 |
| split-K | 1, or slices whose float partials a second kernel sums |

A config is checked against these limits before it builds. A valid one becomes a `cuda` kernel
file, must match PyTorch on an odd-sized `ci` shape and on the timed shape, and is then timed.
Each config, and each final A/B run, gets a fresh worker.

## Why the baseline is not CUTLASS's default

On sm_120 (RTX 50 series), CUTLASS's SM120 builders take only fp8, fp6 and fp4, so fp16 and bf16
GEMMs run the SM80 kernels. Their default config (128x256x64 tile, 3 stages) needs 144 KB of
shared memory and does not launch where a block gets 99 KB. The baseline is that tile with as
many stages as fit.

## Output

`results/tilesearch.csv`, one row per shape: cuBLAS time, the default and its time, the best
config, its time and TFLOPS, the speedups over both, and the cost. Every config tried, with its
status and time, is in `.smforge/runs/<run>/tilesearch/<shape>/trials.json`.

| Option | Default | Meaning |
|---|---|---|
| `--shapes` | required | comma-separated shapes |
| `--rounds` | 3 | proposal rounds per shape |
| `--per-round` | 6 | configs per round |
| `--effort`, `--budget-usd`, `--backend`, `--claude-bin` | | as for the [pipeline](optimization.md#options) |
