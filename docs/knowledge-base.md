# Knowledge base

The knowledge base holds the rules and optimization patterns the LLM stages see:
`knowledge_base/triton/` (21 constraints, 21 patterns) and `knowledge_base/cuda/` (10
constraints, 5 patterns). Each stage's prompt gets the constraints and the patterns for that
stage and the problem's family, filtered for the GPU's architecture and probed features.

## Entries are verified, not written down

Every entry comes with a kernel pair on a `dev` problem, run on the GPU:

- A **constraint** is a rule. Its `checks/<id>/wrong.py` must fail the way the entry says (a
  static check, an error, a mismatch, or slower) and `correct.py` must pass.
- A **pattern** is an optimization. Its `examples/<id>/after.py` must pass the candidate checks
  and beat `before.py` in an interleaved A/B. The measured speedup goes to
  `examples/measured.yaml` and stays out of the prompts.

An entry that does not hold on the GPU stays out. For example, warp-shuffle reductions measured
at parity on RMSNorm and cumsum (both memory-bound at the corpus shapes), so they are not a CUDA
pattern.

## Commands

```bash
smforge kb list                              # entries, stages, measured speedups
smforge kb list --dsl cuda
smforge kb show fusion --families gemm       # a stage's slice as a prompt receives it
smforge kb verify --write                    # run every pair on this GPU, record speedups
smforge kb verify --dsl cuda --ids cutlass_sm80_gemm
pytest runners/test_kb_examples.py           # the same checks, both knowledge bases, as tests
```

## Layout

```
knowledge_base/<language>/
  constraints.yaml                    # rule, wrong and correct snippets, how wrong.py must fail
  patterns/<stage>.yaml               # summary, how, families, example id
  checks/<constraint>/{wrong.py,correct.py}
  examples/index.yaml                 # example id, problem, description
  examples/<example>/{before.py,after.py}
  examples/measured.yaml              # written by kb verify --write
```
