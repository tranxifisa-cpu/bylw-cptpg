# Dense E4 rerun / resume

This code-only update leaves the manuscript unchanged and adds the finalized dense E4 robustness sweep.

## Frozen design

- Reference grid: `eta+`, `eta-` in `{0.00, 0.05, ..., 0.40}` (81 configurations).
- Horizon grid: `h in {1,2,4,5,10,20,25,50}`.
- Optimizer grid: `gamma in {0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12}` and `vartheta in {0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.06}`, restricted by `gamma * 0.31 <= vartheta` (48 configurations).
- Four paired seeds (`2000`--`2003`).
- Primary display quantity: configuration-matched post-change DLR ratio `DLR_online / DLR_frozen`. It is not a new theoretical loss; it is a within-configuration normalization of the same DLR.

The 11-point horizon design contains 560 seed-level jobs. The previous 548-row dense E4 result supplies 540 reusable rows; only the five new horizons at four seeds each must run.

## Recommended run

From this directory:

```bash
python run_e4_dense_incremental.py \
  --output /path/to/e4_dense_results \
  --reuse /path/to/previous/e4_final/seed_metrics.csv \
  --resume-from /path/to/new_seed_metrics_partial.csv \
  --seeds 4 --workers 8
```

Before running, inspect the plan:

```bash
python run_e4_dense_incremental.py \
  --output /path/to/e4_dense_results \
  --reuse /path/to/previous/e4_final/seed_metrics.csv \
  --resume-from /path/to/new_seed_metrics_partial.csv \
  --seeds 4 --workers 8 --plan-only
```

The runner checkpoints after every completed job, so it can be resumed after an interrupted session.

After completion:

```bash
python analyze_e4_dense.py \
  --results /path/to/e4_dense_results \
  --figures /path/to/figures \
  --tables /path/to/tables
```

The analyzer produces a reference-rate DLR-ratio surface, a dense horizon curve with paired-seed bootstrap intervals, and an admissible optimizer-grid heatmap.
