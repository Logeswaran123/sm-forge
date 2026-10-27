# sm-forge: build plan

sm-forge takes a correct Triton kernel and a shape spec, and returns a faster kernel that passes
the same correctness checks on an NVIDIA GPU. An LLM works through a sequence of optimization
stages. Every candidate is compiled, checked against a reference and timed on the GPU before it
is kept. The design follows Xe-Forge (IntelLabs, arXiv:2605.26118), which does the same for Intel
Arc GPUs.

The first target is the GeForce RTX 5080 in this machine (sm_120). The knowledge base is gated by
compute capability, so sm_80, sm_89, sm_90 and sm_100 parts can be added without forking the code.

## 1. What Xe-Forge does

Paper: *Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization for Intel GPU*, Spoczynski et al.,
April 2026. Repository: github.com/IntelLabs/Xe-Forge, v0.4.0, Apache-2.0, about 18k lines of
Python under `src/xe_forge`.

### Pipeline

1. The analyzer (an LLM) reads the kernel and returns typed issues from a taxonomy of 30+
   `IssueType`s (`dtype_float64`, `unfused_elementwise`, `missing_autotune`, `open_ended`, ...).
   Each issue has a severity from 1 to 5, a suggested fix and a speedup estimate.
2. A fixed table routes each issue type to one stage. Stages with no issues are skipped.
3. The planner (an LLM) orders the remaining stages. Its output is filtered, then corrected
   against hard dependencies in `planner.py`: `algorithmic < dtype_fix < fusion < device_specific
   < autotuning`, `memory_access < tensor_descriptors < device_specific`, `discovery < dtype_fix`.
   The fallback order is algorithmic, discovery, dtype_fix, fusion, memory_access,
   tensor_descriptors, persistent_kernel, device_specific, autotuning. (`tensor_descriptors` was
   `block_pointers` until Triton deprecated `tl.make_block_ptr`.)
4. Each stage runs a CoVeR agent (Chain of Verification and Refinement, `agents/cover.py`). The
   LLM writes a thought and a full kernel. The harness runs the verification tool on it. On
   failure the error is appended to the trajectory and the LLM tries again, up to T iterations
   (5 in the paper, 3 in the reproduction config). When the budget runs out, a fallback
   extractor produces one more candidate. It must pass the same checks, or the stage returns its
   input unchanged. On context overflow the oldest iteration is dropped.
5. The kernel is re-analyzed after each stage. The whole sequence can run k times and keep the
   fastest result (best-of-k, default 1).

### Verification cascade

1. Syntax: `ast.parse`.
2. Structure: imports, `@triton.jit`, `class Model`, power-of-two `num_warps`, block dims ≤ 256.
3. Correctness: seeded init, weights copied with `load_state_dict`, cloned inputs, `torch.allclose`
   at rtol 1e-2 and atol 1e-5. NaN is rejected. Inf is rejected unless the original has it.
4. Performance: the candidate must be faster than its input.

### Timing

AI-bench: 200 warmup iterations, 100 timed, a 256 MB L2 flush before each, device events, min and
max trimmed, mean reported. The repository later added interleaved A/B rounds with alternating
order and a 2% minimum-effect band, after an A/A check measured about 1% between identical builds
(`core/executor.py: time_interleaved`).

### Knowledge base

YAML files of constraints (hard rules with a severity and wrong/correct snippets) and patterns
(before/after transformations tagged by stage), plus 19 unoptimized/optimized example pairs used
as few-shot material. The paper counts 84 entries. Each stage's prompt gets the entries for that
stage. The README says the examples are drawn from KernelBench, the suite the paper reports on.

### Engines and tooling

- DSPy engine: the automated pipeline above.
- Coordinator: a ReAct agent that picks stages itself (`agents/coordinator.py`).
- Claude Code engine: writes a workspace (`CLAUDE.md`, `config.yaml`, KB, slash commands) and
  drives it through the `xe-forge-skill` CLI (`analyze`, `validate`, `benchmark`,
  `trial init|save|status|finalize`, `profile`). The skill CLI is the only permitted source of
  timings, and results go into a trial tree (`core/trial_manager.py`).
- Tile search: LLM-proposed tile shapes for CUTLASS-SYCL GEMM and Flash Attention templates.
- Host integration: `--kernel-repo`, `--external-benchmark`, `--lessons`, `--dataset-record`.

### Results (Arc Pro B70, about 160 TFLOPS FP16, 608 GB/s, GPT-5.4)

- 1.17x geomean over PyTorch eager on 97 KernelBench Level 2 kernels. 67% improved, 9 above 5x,
  maximum 82x.
- Flash Attention forward: 2 to 13.3x over the seed kernel on 16 shapes, no regressions.
- Conv and ConvTranspose families regress (0.5 to 0.8x on several).
- Seeds came from KernelFalcon (KernelAgent).
- Typical cost: 10 to 20 LLM calls per kernel, $0.50 to $2.00.
- The paper attributes results above peak to FLOP reduction, not throughput. It names per-stage
  rollback as the most useful next step.

### NVIDIA support in Xe-Forge today

`--device cuda` and `uv sync --extra nvidia` exist, but:

- `knowledge_base/triton/cuda/cuda_optimizations.yaml` contains `patterns: []` and
  `constraints: []`.
- `core/device_query.py: _set_cuda_recommendations` branches on `major >= 9` and returns Hopper
  settings (128x256x64 tiles, 8 warps). sm_120 has major 12, so the RTX 5080 gets Hopper tiles.
  This part has 99 KB of shared memory per block and no wgmma.
- Prompts, the validator and the stage list are XPU-specific: `grf_mode`, 32 warps by default,
  tensor descriptors described as 2D block IO.
- Candidates are compiled and run inside the pipeline process. On CUDA an illegal address is a
  sticky error: every later CUDA call in that process fails, so one bad candidate ends the run.

The parts to carry over:

- the structure: analyze, route, plan, CoVeR per stage, re-analyze;
- the verification cascade and the interleaved timing;
- the oracle rules in `docs/CORRECTNESS.md` and the trial ranking;
- the rule that only the harness produces timings.

The hardware knowledge has to be written from scratch.

## 2. Target hardware

Measured on this machine with `torch.cuda.get_device_properties` and `nvidia-smi` (tessera env:
torch 2.11.0+cu128, triton 3.6.0; CUDA 12.8 toolkit with ncu, nsys, compute-sanitizer, nvdisasm):

| Property | Value |
|---|---|
| GPU | GeForce RTX 5080, compute capability 12.0 (sm_120) |
| SMs | 84 |
| Memory | 16 GB GDDR7 |
| L2 | 64 MB |
| Shared memory | 100 KB per SM, 99 KB (101376 B) per block opt-in |
| Registers | 64K 32-bit per SM |
| Max SM clock | 3090 MHz |
| Driver | 617.42, Windows host, WSL2 guest |

Measured in stage 0 (`smforge device --all`, saved to `hardware/rtx5080.yaml`):

| Item | Result |
|---|---|
| fp16, bf16, tf32 `tl.dot` | HMMA |
| fp8 `tl.dot`, `tl.dot_scaled` (e4m3) | QMMA |
| `num_stages > 1` | cp.async (LDGSTS) |
| Tensor descriptors, host and device side | TMA (UTMALDG, UTMASTG) |
| `tl.range(..., warp_specialize=True)` | compiler error, `PassManager::run failed` |
| DRAM bandwidth | 885 GB/s read, 755 GB/s copy |
| cuBLAS peak | fp16 116, bf16 109, tf32 53, fp32 38, fp8 234 TFLOPS |
| A/A timing noise (p20) | under 1%; suggested min effect 1.7% |
| nsys | works |
| ncu | blocked: `ERR_NVGPUCTRPERM`, counters not enabled on the Windows side |
| compute-sanitizer | blocked: WDDM debugger interface not enabled (`EnableDebuggerInterface.bat`) |

Cluster launch and clock locking were not tested. Clock locking needs root.

## 3. Intel to NVIDIA mapping

| Xe-Forge (Intel Xe) | sm-forge (NVIDIA) |
|---|---|
| XMX / DPAS | Tensor cores: mma.sync (sm_80+), wgmma (sm_90a), tcgen05 (sm_100a) |
| Subgroup | Warp (32 lanes) |
| SLM | Shared memory, budget per block from the device query |
| GRF mode 128/256 | Register budget: `maxnreg`, `num_warps`, occupancy |
| 2D block IO via tensor descriptors | TMA via tensor descriptors (sm_90+); `cp.async` pipelining via `num_stages` (sm_80+) |
| 32 warps by default | Usually 4 or 8 warps, tuned |
| Xe cores | SMs; persistent grid = SM count × CTAs per SM |
| `xpu-smi`, `torch.xpu` | NVML, `nvidia-smi`, `torch.cuda` |
| VTune, unitrace | Nsight Compute (`ncu`), Nsight Systems (`nsys`) |
| SPIR-V check (DPAS vs FMA count) | SASS check (tensor-core MMA vs FFMA, LDGSTS, UTMALDG, spills) |
| CUTLASS-SYCL tile search | CUTLASS / CuTe tile search |

## 4. Design decisions

1. **Triton first.** CUDA C++ comes in stage 7. Python 3.11+, uv, torch cu128, Triton pinned
   (3.6.0 already works with this GPU in the tessera env).

2. **Process isolation.** Every candidate compiles and runs in a worker subprocess with a timeout.
   A worker that hits a CUDA error is discarded and a new one started. Reference outputs for each
   kernel and input draw are computed once and stored as `.pt`, so checking a candidate costs one
   worker run.

3. **Fixed harness and a narrow edit surface.** The LLM returns a kernel file. The harness, the
   spec and the reference are read-only. The paper describes generated kernels that call PyTorch
   through dynamic imports to pass checks. Three guards against that:
   - AST rule: `Model.forward` and its helpers may call only allocation, view and layout ops from
     torch. The allowlist can be extended per kernel.
   - Launch audit: `torch.profiler` records the CUDA kernels one forward launches. A candidate must
     launch at least one Triton kernel. It may not launch a vendor kernel (cuBLAS, cuDNN, aten
     CUTLASS) that the baseline does not launch.
   - Output caching: correctness runs two input draws in the same process. A module that returns
     a stored result fails the second draw.

4. **Facts before the LLM.** The analyzer gets numbers measured from the compiled kernel:
   - registers, spills and shared memory per kernel (`CompiledKernel.n_regs`, `.n_spills`,
     `.metadata.shared`);
   - grid size and waves on 84 SMs;
   - SASS instruction mix (tensor-core MMA vs FFMA, cp.async, TMA), from `.asm["cubin"]` through
     nvdisasm;
   - resident CTAs per SM from the driver's occupancy calculator;
   - arithmetic intensity from the spec's flop and byte formulas, against the measured ridge point;
   - an ncu Speed of Light summary, when counters are available.

   The LLM classifies issues from these instead of guessing from source.

5. **Arch-gated knowledge base.** Each entry has `arch: {min: 80, max: null}` and optional
   `requires: [tma, fp8_mma, ...]`. The loader filters by the detected device and its feature
   probes. Each entry has a runnable before/after pair and a speedup measured on a named GPU, and
   CI runs the pairs.

6. **Per-stage rollback.** A stage result is kept only if it passes correctness and beats the
   stage input by more than the min-effect band in an interleaved A/B measurement. Otherwise the
   stage input carries forward. A regression is never finalized.

7. **LLM layer.** Call the Anthropic Python SDK directly, behind a small `LLMClient` interface. Do
   not use DSPy. CoVeR is a loop of about 200 lines, and calling the SDK directly keeps prompt
   caching, effort and output schemas under our control.
   - Default model `claude-opus-5-5` for every role. Effort is set per role: analyzer `high`,
     optimizer `high`, planner `low`, fallback extractor `medium`.
   - The analyzer and planner use structured outputs (`client.messages.parse` with Pydantic
     models).
   - The optimizer returns the kernel as a fenced code block. No tool calling is needed, because
     the harness verifies every candidate itself.
   - Two API constraints differ from Xe-Forge's config. The Messages API has rejected
     `temperature` since 2026-10-21 (Xe-Forge sets `LLM_TEMPERATURE=1`). Opus 5.5 rejects forced
     `tool_choice`.
   - Prompt layout for caching: a stable prefix (role instructions, device facts, the stage's KB
     slice) marked with `cache_control`, then the kernel and the trajectory.
   - Log `usage` on every call, including `cache_read_input_tokens`. Enforce a per-kernel dollar
     cap.

8. **Two engines.** A pipeline engine (stages 4 and 5) and a Claude Code workspace engine
   (stage 6). Both go through the same harness via `smforge-skill`.

9. **Corpus split.** KB authoring and prompt tuning see only the dev split. The eval split is
   frozen before stage 3 starts.

10. **Timing statistic.** A run's time is a low quantile of its samples (p20 by default), not the
    median. Interference from other work on the GPU only adds time. On this WSL2 machine the
    Windows desktop slowed up to 40% of the samples of a 1.2 ms matmul: the medians of two
    identical A/A arms differed by 16%, their p20s by under 0.1%.

## 5. Repository layout

```
sm-forge/
  pyproject.toml              # uv, hatchling; extras: cu128, dev
  .env.example
  src/smforge/
    cli.py                    # smforge: optimize, eval, device, seed
    skill_cli.py              # smforge-skill: analyze, validate, check, bench, facts, trial, profile
    config.py
    models.py                 # pydantic: Issue, Analysis, StageResult, Trial, RunReport
    spec.py                   # spec.yaml loader, flop/bytes formulas, variants
    render.py                 # text output for the CLIs
    device/
      query.py                # torch, driver and nvidia-smi properties
      probes.py               # feature probes: compile, check SASS and output
      kernels.py              # probe and bandwidth kernels
      roofline.py             # measured bandwidth and matmul peaks
      noise.py                # A/A timing noise
      tools.py                # nvidia-smi; ncu, nsys, compute-sanitizer checks
      profile.py              # hardware/<gpu>.yaml
    harness/
      runner.py               # parent side: worker lifecycle, timeouts, restart after CUDA errors
      worker.py               # subprocess entry: build, run, check, time
      workload.py             # kernel loading, inputs and Model construction per variant
      source.py               # AST view of a kernel file
      static.py               # Triton static validator
      audit.py                # forward-path rules, launch audit
      tolerance.py
      correctness.py          # output comparison, weight copy, reference cache
      timing.py               # events, L2 flush, quantile, interleaved A/B, verdict
      launches.py             # kernels one forward launches, from the profiler
      facts.py                # regs, spills, smem, SASS mix, occupancy, waves
      ncu.py                  # stage 6: ncu / nsys runners and parsers
      sanitizer.py            # stage 6: compute-sanitizer wrapper
    kb/
      schema.py
      loader.py               # arch filter, stage slices, token budget
    llm/
      client.py               # SDK wrapper: caching, structured output, usage, cost cap, replay
      prompts/                # jinja templates per role and stage
    agents/
      analyzer.py
      planner.py
      cover.py
      stages.py               # stage registry, routing table, dependencies
    pipeline.py
    trials.py                 # trial tree
    lessons.py
    claude_ws/                # Claude Code workspace generator and templates
    dsl/
      triton.py
      cuda_cpp.py             # stage 7
    tile_search/              # stage 7
  knowledge_base/
    triton/common/*.yaml
    triton/nvidia/*.yaml
    triton/nvidia/examples/{index.yaml,*.py}
    cuda/nvidia/*.yaml        # stage 7
  corpus/
    <id>/{reference.py,kernel.py,spec.yaml}
    splits.yaml               # dev / eval
  hardware/
    rtx5080.yaml              # stage 0 output
  runners/
    test_kb_examples.py
  scripts/                    # roofline plots, report generation
  tests/
  docs/
```

## 6. Stages

Each stage lands as several focused commits. Exit criteria are checked before the next stage
starts.

### Stage 0: environment and hardware baseline

- `pyproject.toml` with uv. Pin torch 2.11.0+cu128 and triton 3.6.0.
- `smforge device` writes `hardware/rtx5080.yaml`: the properties in section 2, plus feature probes.
  Each probe compiles a small Triton kernel and checks its SASS. The probes cover:
  - tensor-core MMA for fp16/bf16 `tl.dot`;
  - fp8 `tl.dot`;
  - TMA from `tl.make_tensor_descriptor`;
  - `tl.dot_scaled`;
  - warp specialization.
- Measured roofline:
  - DRAM bandwidth from copy and read-only kernels at sizes above L2, and L2 bandwidth;
  - tensor throughput per dtype from a cuBLAS/cuBLASLt GEMM sweep (fp16, bf16, tf32, fp8 if
    available);
  - FP32 FMA peak.

  Store ridge points. These replace spec-sheet numbers everywhere downstream.
- Timing noise:
  - Run an A/A test (the same kernel in both arms) on a small kernel, a memory-bound kernel and a
    compute-bound kernel, using stage 1's timing code. Set `min_effect` from the spread.
  - Check whether `nvidia-smi -lgc` works under WSL2.
  - Measure the effect of the Windows desktop sharing the GPU.
- Profiler access:
  - Run `ncu --section SpeedOfLight` on a toy kernel. Under WSL2 this needs GPU performance
    counter access enabled on the Windows side.
  - Check that `nsys` and `compute-sanitizer` run.

Exit: `hardware/rtx5080.yaml` committed with measured peaks and probe results. The A/A spread is
known. There is a written list of what works under WSL2 and what does not.

### Stage 1: harness, no LLM

Kernel contract, KernelBench-compatible and the same as Xe-Forge's:

- `kernel.py`: `class Model(nn.Module)`, `get_inputs()`, `get_init_inputs()`;
- `reference.py`: the PyTorch `Model`;
- `spec.yaml`: `inputs`, `inits`, and variants (`ci`, `bench-gpu`, ...) with `dims`, `dtype`,
  `flop`, `bytes`, and optional `rtol`/`atol`.

Xe-Forge's test kernels then port with device changes only.

Work:

- Spec loader with formula evaluation.
- Worker subprocess and pool: JSON over pipes, timeout, restart after a CUDA error.
- Correctness:
  - seeds, weight copy, cloned inputs, two input draws;
  - tolerances by dtype; precedence CLI > spec > env > default;
  - NaN and Inf rules;
  - diagnostics: max abs/rel error, share of mismatched elements, first bad index.
- Timing:
  - device events, a 256 MB flush buffer (4x L2), and a dummy matmul queued ahead of the start
    event so launch latency is not timed (as Xe-Forge does);
  - warmup, trimmed statistics;
  - interleaved A/B rounds with alternating order, medians and per-round ratios;
  - a verdict of `faster`, `slower` or `parity` against `min_effect`;
  - an optional CUDA graph mode for launch-bound kernels, reported separately.
- Two time measures per run: forward wall time (events around `forward`) and kernel-only time (sum
  of device kernel durations from the profiler). Many seeds cast weights or synchronize inside
  `forward`. Removing that is a real speedup, but it is not a kernel speedup, and reporting both
  keeps the difference visible.
- Static validator:
  - syntax, imports, `@triton.jit`, `class Model`;
  - power-of-two `num_warps` and `BLOCK_*`;
  - `tl.dot` operand dims ≥ 16;
  - estimated shared memory per config ≤ device opt-in;
  - no `.item()`, `.cpu()` or `torch.cuda.synchronize()` in `forward`.
- Audit: the AST rule and launch audit from section 4.
- Facts: regs, spills, smem, SASS mix, waves, arithmetic intensity.
- CLI: `smforge-skill validate | check | bench | facts`.
- Tests:
  - unit tests per module;
  - integration tests on hand-written kernels, including broken ones: wrong mask, out-of-bounds
    write (must kill the worker and recover), a call to `torch.matmul`, an output cache, a host
    sync in `forward`.

Exit: hand-written kernels repeat within the A/A band across runs. Every broken kernel is rejected
with a message that names the cause.

### Stage 2: corpus and baselines

- Port KernelBench Level 2 (all 100) and a selection of Level 1 into the corpus format, with
  fp16/bf16 variants, and fp32 where the problem needs it. Port Xe-Forge's `test_kernels/` and
  `examples/vllm/`, replacing `torch.xpu` with `torch.cuda`.
- Add inference kernels:
  - RMSNorm, fused add + RMSNorm, LayerNorm, RoPE, SwiGLU, softmax;
  - Flash Attention forward (prefill shapes), decode and paged attention;
  - fused MoE;
  - FP8 GEMM with per-tensor and per-row scales.

  Use real model shapes sized for 16 GB. Check the license of any kernel imported from vLLM or
  SGLang before committing it.
- Seed kernels. Pick one of:
  - Option A: KernelAgent (meta-pytorch/KernelAgent) on the reference, as the paper did.
  - Option B: `smforge seed`, which runs the CoVeR loop with correctness-only verification and a
    prompt asking for the straightforward kernel.

  Freeze the seeds. A seed in the eval split is never regenerated after results are reported.
- Baselines per variant: eager, `torch.compile` (default and `max-autotune`), seed kernel. Write
  `results/baselines.csv` with torch, triton and driver versions and the git hash.
- Split `corpus/splits.yaml` into dev and eval, stratified by family (GEMM, matmul, conv,
  convT, norm, attention, MoE, reduction). Freeze the eval split.
- Roofline script: each kernel plotted at its arithmetic intensity against the stage 0 ceilings.

Exit: at least 120 problems with passing seeds and measured baselines. The split is committed.

### Stage 3: knowledge base

Schema (YAML, validated with pydantic):

```yaml
constraints:
  - id: smem_budget_per_config
    severity: critical
    arch: {min: 80}
    stages: [device_specific, autotuning]
    rule: "num_stages * (BM*BK + BK*BN) * elem_bytes <= shared_memory_per_block_optin"
    wrong: |
      triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_stages=4)
    correct: |
      # prune_configs_by drops configs over the device's smem budget
patterns:
  - id: gemm_group_m_swizzle
    stage: device_specific
    arch: {min: 80}
    requires: []
    applies_to: [gemm, matmul, bmm]
    before: examples/gemm_noswizzle.py
    after: examples/gemm_swizzle.py
    rationale: L2 reuse across CTAs launched together
    measured:
      - {gpu: rtx5080, variant: bench-gpu, speedup: null}
```

`measured` is written by `runners/test_kb_examples.py`, never by hand. Prompts do not show
speedups. Xe-Forge's Claude template tells the agent to ignore them, and this KB leaves them out.

First content. Each entry is verified on the RTX 5080 before it goes in.

Constraints:

- Shared memory per config within the device's opt-in limit (99 KB per block here).
- `tl.dot` operand shapes ≥ 16 and powers of two.
- fp32 `tl.dot` defaults to TF32 (`input_precision`). Against an fp32 reference with tight
  tolerance, use `input_precision="ieee"` or change the dtype policy explicitly. PyTorch's matmul
  does not use TF32 by default.
- int64 offsets when `numel` can exceed 2^31.
- Register budget: `num_warps * 32 * regs_per_thread ≤ 65536`. Spills are reported by facts.
- No host syncs, `.item()` or `.cpu()` in `forward`. No per-call weight casts or transposes: do them
  once and cache them on the module.
- Tensor descriptors (TMA): 16-byte aligned base, non-last strides that are multiples of 16 bytes.
- Autotune: `key` must include the shape arguments that change the best config. Kernels that
  accumulate in place need `reset_to_zero` or `restore_value`.
- fp32 `atomic_add` makes results order-dependent; check against the tolerance.
- GROUP_M swizzling needs a 1D grid.

Patterns by stage:

| Stage | Patterns |
|---|---|
| algorithmic | algebraic rewrites (reduce before a GEMM when the reduction is linear, fold constant scales); review and reuse Xe-Forge's `common/algorithmic_patterns.yaml` |
| dtype_fix | fp16/bf16 I/O with fp32 accumulate; fp8 e4m3 with scales only when the spec tolerances allow; bf16 when fp16 range is at risk |
| fusion | GEMM epilogue fusion (bias, activation, scale, residual); a reduction in the GEMM epilogue when N fits one tile; online softmax; no fusion that pushes registers into spills |
| memory_access | contiguous dim on the fastest axis; `tl.multiple_of` / `tl.max_contiguous` for vector width; masks only on tails (`EVEN_K` specialization); `eviction_policy`; contiguity in the wrapper; channels-last for conv |
| tensor_descriptors | TMA loads via `tl.make_tensor_descriptor` on sm_90+, and on sm_120 if the probe passes; `num_stages` pipelining on sm_80+ |
| persistent_kernel | persistent GEMM with `NUM_SMS` from the device; split-K / stream-K for small M·N with large K; tail-wave analysis (tile count mod SM count) |
| device_specific | tile shape tables per arch from measured sweeps; `num_warps`, `num_stages`, `maxnreg`; GROUP_SIZE_M; warp specialization where the probe passes |
| autotuning | config space generated from the constraints and pruned by smem and register budget (`prune_configs_by`); at most 12 configs; correct keys |

Examples: 15 to 20 unoptimized/optimized pairs, taken from the dev split only.

Loader: filter by arch and probes, slice by stage, rank entries by how well their `applies_to`
matches the kernel's op families, and cap each stage at a token budget (start at 6k).

Exit: at least 60 entries, each with a before/after test passing on the RTX 5080.
`runners/test_kb_examples.py` passes.

### Stage 4: LLM client and CoVeR

- `llm/client.py`:
  - model and effort per role from config;
  - streaming for long outputs;
  - structured outputs via `messages.parse`;
  - prompt caching on the stable prefix, with `cache_read_input_tokens` checked so a silent miss
    shows up in the log;
  - per-call cost log and a per-kernel cost cap;
  - SDK retries for 429 and 5xx;
  - handling of `stop_reason == "refusal"`.
- Record and replay: every request and response stored as JSONL keyed by prompt hash. Tests run
  from recordings without network access.
- CoVeR:
  - each iteration produces a thought and a full kernel;
  - verification order: syntax, static, audit, compile in a worker, correctness, facts, timing
    against the stage input;
  - feedback is short: the first compiler error with its source line, correctness diagnostics,
    the timing verdict, fact deltas (`regs 96 -> 168, spills 0 -> 24`);
  - T = 3 by default, configurable;
  - fallback extractor;
  - truncation by dropping the oldest iterations;
  - every failed candidate saved with stage, iteration and error class.
- Output parsing takes the last fenced Python block. The candidate is rejected if it changes
  `get_inputs`, `get_init_inputs` or the `Model.__init__` signature.
- First stage to wire up: autotuning, which has the narrowest edit surface.

Exit: the autotuning stage runs over the whole dev split with no correctness failures in finalized
kernels. Replay tests pass offline. Accept rate and cost per kernel are recorded.

### Stage 5: analyzer, planner, all stages

- Issue taxonomy. Start from Xe-Forge's `IssueType` and drop the XPU-only types (`missing_grf_mode`,
  `missing_packed_transpose`). Add:

  | Issue | Detected from |
  |---|---|
  | `no_tensor_core_mma` | SASS shows an FFMA loop where a dot was intended |
  | `tf32_precision_risk` | fp32 dot with default precision against a tight tolerance |
  | `register_spill` | `n_spills > 0` |
  | `low_occupancy` | regs or smem limit CTAs per SM |
  | `smem_over_budget` | config estimate above the device limit |
  | `missing_pipelining` | `num_stages = 1`, no cp.async in SASS |
  | `missing_tma` | sm_90+ or probe-enabled sm_120, no UTMALDG |
  | `tail_wave` | small remainder of tiles mod SMs |
  | `grid_underfill` | fewer CTAs than SMs |
  | `split_k_opportunity` | small M·N, large K |
  | `launch_bound` | many short kernels per forward |
  | `uncoalesced_access` | ncu memory sectors per request, or the analyzer |
  | `bank_conflict` | ncu shared-memory metrics |
  | `fp8_opportunity` | spec tolerance allows it, compute-bound GEMM |
  | `host_sync_in_forward`, `per_call_weight_prep` | AST |

- Routing table from issue to stage, with a test that every issue type routes somewhere.
- Analyzer: structured output. Input is the kernel, the reference, the spec summary, the facts and
  the roofline position. A deterministic pre-analyzer emits the fact-backed issues (spills, waves,
  FFMA dot, host sync), and the LLM cannot drop them.
- Planner: LLM ordering, Xe-Forge's hard-dependency correction, skip logic, deterministic fallback.
- One prompt template per stage, with that stage's KB slice.
- Re-analysis after each stage, per-stage rollback, best-of-k.
- Discovery stage for `open_ended` proposals. A proposal needs a validity argument and a speedup
  estimate, and goes through the same verification.

Exit: the full pipeline runs over the dev split, and no regression is finalized. The run report
has accept rates and cost per stage.

### Stage 6: trials, profiling, Claude Code engine

- Trial tree as in Xe-Forge's `trial_manager.py`:
  - `t0` is the baseline; children record their parent and a sha256 of their source;
  - ranking: correct first, then speedup; a measured result beats an unmeasured one; parity ranks
    at 1.0;
  - `finalize` refuses regressions.
- ncu:
  - profile a single launch (`--kernel-name`, `--launch-skip`);
  - sections: SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, LaunchStats, WarpStateStats,
    SourceCounters;
  - parse the CSV into about 30 lines for the prompt, and diff parent against child.
- nsys for multi-kernel forwards: gaps and launch overhead.
- compute-sanitizer (memcheck, racecheck): run it on candidates that crash or mismatch
  nondeterministically, and on every finalized winner.
- Claude Code workspace generator:
  - contents: `CLAUDE.md`, `config.yaml`, the KB filtered for the device, an `optimize-kernel`
    slash command;
  - rules: only trial files may be written; only `smforge-skill bench` produces timings; the
    oracle is read-only;
  - headless runs through the Claude Agent SDK or `claude -p`, with a turn cap.
- Lessons ledger: a markdown file per kernel recording what was tried and measured. Later runs on
  the same GPU read it.

Exit: the Claude Code engine optimizes a dev kernel unattended and finalizes a winner. The pipeline
harness re-measures that winner within the A/A band.

### Stage 7: CUDA C++ and CUTLASS tile search

- DSL registry: `triton`, `cuda`.
- CUDA C++ kernels:
  - built as torch extensions with `torch.utils.cpp_extension.load_inline` (ninja,
    `-arch=sm_120`) and cached by source hash;
  - compile errors parsed to file:line for feedback;
  - facts from cuobjdump/nvdisasm and cuda-breakdown;
  - validator for launch bounds, bounds checks, and `cudaGetLastError` after launches.
- KB `cuda/nvidia/`: vectorized loads, shared-memory tiling with padding against bank conflicts,
  `cp.async`, warp-shuffle reductions, `mma.sync` via inline PTX or WMMA, `__launch_bounds__`.
- Tile search:
  - CUTLASS GEMM and Flash Attention templates (Jinja) parameterized by tile shape, stages and
    warp layout;
  - the LLM proposes configs within the smem and register limits, then a
    propose-validate-benchmark loop runs, with cuBLAS as the reference point;
  - first, check which CUTLASS kernels target sm_120, and whether the CuTe Python DSL supports
    sm_120 (it would avoid long C++ builds).

Exit: CUDA C++ seeds go through the same pipeline. Tile search beats CUTLASS's default config on a
set of GEMM shapes.

### Stage 8: evaluation

- `smforge eval --split eval` runs the pipeline over the eval split with a fixed config. It writes
  one JSON file per kernel and a summary.
- Metrics:
  - geomean speedup over eager, over `torch.compile` max-autotune, and over the seed;
  - share of kernels improved; regressions finalized (target 0);
  - correctness on a fresh input draw and on the `ci` variant shapes, which are not used during
    optimization and so catch configs fitted to one shape;
  - kernel-only and forward speedups, reported separately;
  - fraction of roofline; results above roofline flagged as FLOP reduction;
  - LLM calls, tokens, cache-hit share, dollars and wall time per kernel.
- Ablations on the dev split:
  - no KB; KB without arch gating; no facts; no re-analysis; no rollback;
  - T = 1, 3, 5; one pass vs best-of-3;
  - model: Opus 5.5 at several effort levels, Sonnet 5.5 and Haiku 5.5. This tests the paper's
    claim that domain knowledge matters more than model scale.
- Report with roofline plots. Results committed with library versions, driver version and git hash.

Exit: report on the eval split, plus the ablation table.

### Stage 9: host integration (optional)

- `--kernel-repo` for vLLM/SGLang Triton kernels: locate the kernel, copy it as the baseline, build
  a spec from its call site, emit a patch.
- `--external-benchmark` with Xe-Forge's stdout contract (`CORRECTNESS`, `VERDICT`, and `DONE` as
  the last line).
- CI: lint and CPU unit tests on GitHub Actions. Harness and KB tests on a self-hosted GPU runner.
- A second GPU (sm_89 or sm_90) to exercise arch gating.

## 7. Cost

Opus 5.5 costs $4 per MTok input, $20 per MTok output, and $0.20 per MTok for cache reads. Take a
CoVeR call with a 20k-token prompt (15k of it cached) and 8k output tokens including thinking:

| Part | Tokens | Cost |
|---|---|---|
| Uncached input | 5k | $0.020 |
| Cached input | 15k | $0.003 |
| Output | 8k | $0.160 |
| Total per call | | about $0.18 |

At Xe-Forge's typical 10 to 20 calls per kernel, that is $2 to $4 per kernel. One pass over 100
kernels costs about $300, and best-of-k multiplies it. Stage 4 measures the real figure, and the
per-kernel cap is set from it.

## 8. Risks and open questions

- **WSL2.** ncu counters, clock locking, and timing noise from the Windows desktop on the same GPU.
  If stage 0 shows the noise is too large, run evals under native Linux.
- **Triton on sm_120.** Feature support may lag sm_90 and sm_100. The stage 0 probes decide which
  KB entries are enabled.
- **16 GB of memory.** This limits shapes for long-context attention and large MoE. Size spec
  variants to fit.
- **Seed quality.** Seeds set the headline numbers, and seeds with per-call casts inflate them.
  Report kernel-only time and seed provenance.
- **Benchmark gaming.** Covered by the fixed harness, the AST and launch audits, held-out shapes and
  fresh input draws. Some KernelBench Level 2 problems collapse algebraically, which inflates
  geomeans. Report those separately.
- **Cost.** Opus at high effort over 100+ kernels with best-of-k adds up. Use the cap and caching.
- **KB drift.** KB entries can break with new Triton releases. Pin versions, and rerun the KB tests
  on every upgrade.

## 9. References

- Xe-Forge: https://github.com/IntelLabs/Xe-Forge (read at v0.4.0)
- Spoczynski et al., *Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization for Intel GPU*,
  arXiv:2605.26118
- KernelBench: https://github.com/ScalingIntelligence/KernelBench
- KernelAgent / KernelFalcon: https://github.com/meta-pytorch/KernelAgent
- AI-bench (Xe-Forge's timing harness): https://github.com/libxsmm/AI-bench
- cuda-breakdown: `~/projects/optimizer-tool`
