Optimize `kernel.py` as `CLAUDE.md` describes, in at most $max_trials trials.

1. Read `CLAUDE.md`, `kernel.py`, `problem/reference.py`, `lessons.md`, and the parts of `kb.md`
   and `stages.md` that bear on the issues listed in `CLAUDE.md`.
2. Pick the change most likely to pay off, given the measurements and issues. Write the whole
   kernel file to a new file under `work/`, run `smforge-skill validate` on it, then
   `smforge-skill trial save` with `--parent` and a one-line `--strategy`.
3. Read the result. A failed check says what to fix. A trial at parity or slower is still
   correct: look at its fact changes before trying another direction.
4. Start the next trial from the best one so far (`smforge-skill trial best`), or from an earlier
   one when a direction stops paying off.
5. Stop after $max_trials trials, or sooner when two trials in a row bring nothing. Finish with
   `smforge-skill trial status` and a few lines on what worked and what did not.
