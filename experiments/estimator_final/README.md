# Final estimator experiments

This directory is the repository-integration snapshot for the final estimator study.

Proposed estimator: **centered LOO base + raw independent randomized correction**.
Primary baseline: **CPT-PG v5 adapted raw independent-split plug-in**.

`source/` contains the exact estimator, E2, online-training, variance-decomposition and plotting code together with the frozen 30-stock input panel, frozen independent E2 reference vectors, and the vendor snapshot used in the experiments. `proofs/` contains the proof notes used to audit the estimator construction.

The two new variance-decomposition experiments are:

- A: fixed execution-return path + fixed action/evaluation random numbers; vary only estimator randomness.
- B: fixed estimator/action/evaluation random numbers; vary only the execution-return path.

Use `reproduce.sh` from this directory to rerun E2, the main online comparison, and experiments A/B. All output directories must be absent before a full rerun.
