# Dynamic-Reference CPT-PG Project, v2

This project contains a directly compilable academic LaTeX manuscript and a reproducible experiment folder for

**Dynamic-Reference Cumulative Prospect Policy Gradients: Debiased Estimation, Minimax Limits, and Online Portfolio Tracking**.

The revision follows the opening-report scenario: a nonstationary portfolio-advice setting with five risky assets plus cash, asymmetric dynamic reference-point adaptation, a factor-parametrized Dirichlet policy, a stable real execution path, and a stochastic learning path for CPT policy-gradient estimation.

## Main files

- `main.tex` — manuscript source in English LaTeX.
- `references.bib` — bibliography.
- `main.pdf` — compiled manuscript.
- `figures/` — exported PDF/PNG/SVG figures used by the manuscript.
- `experiments/generate_experiments.py` — deterministic script that regenerates all figures and CSV outputs.
- `experiments/data/ashare_semisynthetic_panel_2023_2026.csv` — deterministic semi-synthetic A-share-style factor panel.
- `experiments/data/README_data.md` — instructions for replacing shipped data with local Tushare/Xueqiu CSV files.
- `experiments/results/` — CSV outputs used by the manuscript.

## What changed in v2

Theoretical additions:

1. CPT score-gradient identity.
2. Bias analysis for the finite-rank plug-in gradient estimator.
3. Multilevel debiased estimator that is exactly unbiased for the regularized CPT gradient.
4. Finite-sample variance bound.
5. Multivariate asymptotic normality.
6. Minimax lower bound for CPT gradient estimation.
7. Dynamic local regret bound with finite-sample estimation error, exogenous market drift, and endogenous reference drift.

Experimental additions:

1. Project-map figure aligned with the opening-report workflow.
2. Path-dependence and disposition-effect diagnostics.
3. Estimator diagnostics for bias, variance, CLT, and minimax scaling.
4. Semi-synthetic A-share-style tracking experiment with baselines.
5. Ablations for dynamic/static and asymmetric/symmetric reference updates.
6. Sensitivity analyses for window length and step size.
7. Heterogeneous retail-agent adoption and satisfaction evaluation.

## Rebuild manuscript

```bash
make
```

The Makefile uses `pdflatex` and `bibtex8`.

## Regenerate experiments

```bash
python experiments/generate_experiments.py
```

For deterministic numerical libraries, optionally run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/generate_experiments.py
```

## Data note

No raw Tushare A-share panel or Xueqiu retail behavior file was included in the supplied attachments. The shipped results therefore use a deterministic semi-synthetic A-share-style market and simulated heterogeneous retail agents. This is deliberate: the experiments validate the theoretical mechanism without fabricating unobserved real-user behavior. To run fully real local data, follow `experiments/data/README_data.md` and replace the shipped panel with local CSV files.
