# E1-E4 aligned experiment implementation

This directory is the implementation used by the final E1-E4 manuscript package.

- E2 defines the shared semi-synthetic factor-block setup and evaluates fixed gradient targets.
- E3 runs the same setup continuously with diminishing-gain endogenous state transitions and no episode reset.
- E4 perturbs reference-update rates, episode horizon, and optimizer scale around the frozen E3 configuration.
- Dynamic local regret (DLR) is the discounted temporal average of projected-residual vectors followed by the squared norm; numerical evaluation uses independent cross streams.

## Dense E4 code update (2026-09-18)

`run_e4_dense_incremental.py` and `analyze_e4_dense.py` implement the finalized dense robustness design. The runner reuses the sparse E4 seed-level results, resumes from a checkpoint CSV, and checkpoints after every completed job. See `E4_DENSE_RUNBOOK.md`.
