# Contributing to sm-forge

Thanks for taking the time to help. Bug reports, fixes, new problems, knowledge base entries,
results from other GPUs and doc improvements are all welcome.

## Before you start

- **Bugs:** open an issue with the command you ran, its full output, your GPU, driver, and
  PyTorch and Triton versions (`smforge device` prints them).
- **Larger changes:** open an issue first to talk it through, so the work doesn't clash with
  something already planned.
- **Security problems:** don't open a public issue. Report them privately from the repository's
  **Security** tab (Report a vulnerability).

## Set up

```bash
git clone https://github.com/Logeswaran123/sm-forge.git
cd sm-forge
uv sync --extra cu128 --extra dev
source .venv/bin/activate
```

The [README](README.md#quick-start) has the requirements and a setup without uv.

## Tests and style

```bash
pytest                                  # GPU tests are skipped without an NVIDIA GPU
pytest runners/test_kb_examples.py      # every knowledge base pair on this GPU
ruff check .
ruff format .
```

- The tests make no LLM calls and need no API key.
- A test that needs a GPU gets `@pytest.mark.gpu`, so it is skipped on machines without one.
- New behavior comes with a test. A bug fix comes with a test that fails without the fix.
- Code is formatted with `ruff format` (line length 100) and must pass `ruff check`. Python 3.11
  or newer.

If you have no NVIDIA GPU, run the CPU tests and say so in the pull request.

## What to contribute

### Fixes and features

Keep each pull request to one change. If it touches the harness (correctness checks, timing, the
launch audit), explain how you know it still rejects what it should: a kernel that is wrong, one
that calls cuBLAS or ATen, one that is slower.

### New problems

Add a problem under `corpus/custom/<id>/` as shown in
[bring your own kernel](docs/examples.md#6-bring-your-own-kernel), with a reference, a spec and a
seed that passes `smforge-skill check`. Don't move existing problems between `dev` and `eval`:
the `eval` split stays frozen so its results stay comparable.

### Knowledge base entries

Every entry is verified on a GPU, so an entry comes with its kernel pair
([knowledge base](docs/knowledge-base.md)):

- a constraint with `checks/<id>/wrong.py` that fails the way the entry says, and `correct.py`
  that passes
- a pattern with `examples/<id>/before.py` and `after.py`, where `after.py` beats `before.py`

Run `smforge kb verify --ids <entry id>` (add `--dsl cuda` for CUDA entries) and paste its
output in the pull request, with the GPU it ran on. An entry that does not hold on the GPU is not
added.

### Results from other GPUs

The committed results all come from one RTX 5080. Numbers from other GPUs are useful, but don't
overwrite the files in `results/`. Open an issue with the command, its output and your
`hardware/<gpu>.yaml` from `smforge device --all`.

### Docs

Fixes to anything unclear are welcome. Keep the writing plain and short, and run any command you
document so its output is real.

## Pull requests

- Branch from `main`. Pull requests are merged into `main`, and direct pushes to it are blocked.
- Split a large change into focused commits. A commit's subject says what changed, for example
  "Check correctness on input draws from first_draw".
- Start the description with what the change does, with real output and a before and after where
  it fits. Then list the limits: what was not tested, and on which GPU it ran.
- Don't commit `.env`, `.smforge/` or results from exploratory runs.

## License

sm-forge is released under [Apache-2.0](LICENSE). By submitting a contribution you agree that it
is licensed under the same terms.
