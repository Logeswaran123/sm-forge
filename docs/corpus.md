# The problem corpus

`corpus/` holds 138 problems: 26 of Level 1 (single ops) and all 100 of Level 2 (short chains
of ops to fuse) of [KernelBench](https://github.com/ScalingIntelligence/KernelBench), and 12
inference kernels (attention, norms, RoPE, MoE, fp8 GEMMs).

Each problem has a family, its main op (`gemm`, `conv2d`, `norm`, ...), which picks the knowledge
base patterns its prompts get. Within each family the problems are split into `dev` (96), which
development used, and `eval` (42), which stays frozen so the final results come from problems the
tool was not tuned on.

Each problem is a directory `corpus/<source>/<id>/`:

| File | What it holds |
|---|---|
| `reference.py` | the PyTorch oracle |
| `spec.yaml` | shapes, dtypes and variants ([kernel format](kernel-format.md)) |
| `problem.yaml` | id, source, family, upstream link, notes |
| `kernel.py`, `seed.yaml` | the Triton seed and how it was written (model, cost, commit) |
| `kernel_cuda.py`, `seed_cuda.yaml` | the CUDA C++ seed, where there is one (14 dev problems) |

A seed is a plain, correct kernel the optimizer starts from. The LLM writes it, and it is kept
only if it passes the static checks, matches the reference on `ci` and `bench-gpu`, and launches
no library or ATen compute kernel.

## Commands

```bash
smforge corpus list --split dev                            # problems, family, split, seed status
smforge corpus import-kernelbench <KernelBench checkout> --level 2
smforge corpus split                                       # assign new problems, eval stays frozen
smforge seed --ids l1_026_gelu                             # write a Triton seed with the LLM
smforge seed --dsl cuda --ids l1_026_gelu                  # a CUDA C++ seed
smforge corpus baselines --modes eager,compile,seed        # times to results/baselines.csv
python scripts/roofline.py                                 # plots/roofline.png
```

`seed` skips problems that already have a seed unless given `--force`.

The `bench-gpu` shapes were sized for an RTX 5080 (16 GB). On a smaller GPU, lower the dims in
`spec.yaml` before running.
