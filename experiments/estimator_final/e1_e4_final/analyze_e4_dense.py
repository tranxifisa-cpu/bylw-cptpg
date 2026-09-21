from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_e4_dense_incremental import REFERENCE_GRID, HORIZON_GRID, GAMMA_GRID, VARTTHETA_GRID, admissible


def bootstrap_mean_ci(values, seed: int, draws: int = 5000) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    reps = np.empty(draws)
    for b in range(draws):
        reps[b] = x[rng.integers(0, n, n)].mean()
    q = np.quantile(reps, [0.025, 0.975])
    return float(q[0]), float(q[1])


def save_figure(fig, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(folder / f"{stem}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    d = seed_metrics.copy()
    d["dlr_ratio"] = d["online_dlr"] / d["frozen_dlr"]
    rows = []
    for i, (keys, x) in enumerate(d.groupby(["panel", "key1", "key2"], dropna=False)):
        lo, hi = bootstrap_mean_ci(x["dlr_ratio"], seed=20260918 + i)
        rows.append({
            "panel": keys[0],
            "key1": float(keys[1]),
            "key2": np.nan if pd.isna(keys[2]) else float(keys[2]),
            "dlr_ratio_mean": float(x["dlr_ratio"].mean()),
            "dlr_ratio_ci_low": lo,
            "dlr_ratio_ci_high": hi,
            "online_dlr_mean": float(x["online_dlr"].mean()),
            "frozen_dlr_mean": float(x["frozen_dlr"].mean()),
            "seeds": int(x["seed"].nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze completed dense E4 robustness sweep using matched post-change DLR ratios.")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)

    seed_path = args.results / "seed_metrics.csv"
    if not seed_path.exists():
        raise FileNotFoundError(f"completed dense result not found: {seed_path}")
    d = pd.read_csv(seed_path)
    s = summarize(d)
    s.to_csv(args.tables / "e4_dense_robustness_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    # A. Reference-adaptation landscape over the actually simulated 9x9 grid.
    ref = s[s.panel == "reference"]
    etas = np.asarray(REFERENCE_GRID, dtype=float)
    z = np.full((len(etas), len(etas)), np.nan)
    for _, row in ref.iterrows():
        ix = int(np.argmin(np.abs(etas - row.key1)))
        iy = int(np.argmin(np.abs(etas - row.key2)))
        z[iy, ix] = row.dlr_ratio_mean
    xx, yy = np.meshgrid(etas, etas)
    levels = np.linspace(np.nanmin(z), np.nanmax(z), 21)
    cf = axes[0].contourf(xx, yy, z, levels=levels)
    if np.nanmin(z) <= 1.0 <= np.nanmax(z):
        axes[0].contour(xx, yy, z, levels=[1.0], colors="black", linewidths=1.0)
    axes[0].plot([etas.min(), etas.max()], [etas.min(), etas.max()], linestyle="--", linewidth=0.9)
    axes[0].plot(0.20, 0.05, marker="*", ms=12, mfc="none", mec="black", mew=1.3)
    axes[0].set_xlabel(r"gain adaptation $\eta_+$")
    axes[0].set_ylabel(r"loss adaptation $\eta_-$")
    axes[0].set_title("A  Reference adaptation", loc="left", fontweight="bold")
    cb0 = fig.colorbar(cf, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label(r"post-change DLR ratio $\mathrm{DLR}_{on}/\mathrm{DLR}_{fr}$")

    # B. Horizon sweep; each point is compared only with its own matched frozen control.
    h = s[s.panel == "horizon"].sort_values("key1")
    x = h.key1.to_numpy(dtype=float)
    y = h.dlr_ratio_mean.to_numpy(dtype=float)
    lo = h.dlr_ratio_ci_low.to_numpy(dtype=float)
    hi = h.dlr_ratio_ci_high.to_numpy(dtype=float)
    axes[1].plot(x, y, marker="o", lw=1.8)
    axes[1].fill_between(x, lo, hi, alpha=0.15)
    axes[1].axhline(1.0, linestyle=":", linewidth=1.0)
    axes[1].axvline(5.0, linestyle=":", linewidth=0.9)
    axes[1].set_xticks(x, [str(int(v)) for v in x])
    axes[1].set_xlim(0, 51)
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].set_xlabel("episode horizon $h$")
    axes[1].set_ylabel(r"post-change DLR ratio $\mathrm{DLR}_{on}/\mathrm{DLR}_{fr}$")
    axes[1].set_title("B  Episode horizon", loc="left", fontweight="bold")
    axes[1].grid(alpha=0.15)

    # C. Dense admissible optimizer grid. Inadmissible combinations remain blank.
    opt = s[s.panel == "optimizer"]
    gs = np.asarray(GAMMA_GRID, dtype=float)
    vs = np.asarray(VARTTHETA_GRID, dtype=float)
    zm = np.ma.masked_all((len(vs), len(gs)), dtype=float)
    for _, row in opt.iterrows():
        ix = int(np.argmin(np.abs(gs - row.key1)))
        iy = int(np.argmin(np.abs(vs - row.key2)))
        zm[iy, ix] = row.dlr_ratio_mean
    im = axes[2].imshow(zm, origin="lower", aspect="auto")
    axes[2].set_xticks(range(len(gs)), [f"{v:g}" for v in gs])
    axes[2].set_yticks(range(len(vs)), [f"{v:g}" for v in vs])
    axes[2].set_xlabel(r"step size $\gamma$")
    axes[2].set_ylabel(r"normalization floor $\vartheta$")
    axes[2].set_title("C  Optimizer scale", loc="left", fontweight="bold")
    ix0 = int(np.argmin(np.abs(gs - 0.08)))
    iy0 = int(np.argmin(np.abs(vs - 0.05)))
    axes[2].plot(ix0, iy0, marker="*", ms=12, mfc="none", mec="black", mew=1.3)
    cb2 = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label(r"post-change DLR ratio $\mathrm{DLR}_{on}/\mathrm{DLR}_{fr}$")

    fig.tight_layout()
    save_figure(fig, args.figures, "e4_robustness_dense")

    def row_for(panel: str, key1: float, key2: float | None):
        q = s[s.panel == panel]
        q = q[np.isclose(q.key1, key1)]
        if key2 is None:
            q = q[q.key2.isna()]
        else:
            q = q[np.isclose(q.key2, key2)]
        return None if q.empty else q.iloc[0].to_dict()

    summary = {
        "quantity": "configuration-matched post-change DLR ratio (online/frozen)",
        "neutral_value": 1.0,
        "reference_configs": int((s.panel == "reference").sum()),
        "reference_below_one": int(((s.panel == "reference") & (s.dlr_ratio_mean < 1.0)).sum()),
        "horizon_configs": int((s.panel == "horizon").sum()),
        "horizon_below_one": int(((s.panel == "horizon") & (s.dlr_ratio_mean < 1.0)).sum()),
        "optimizer_configs": int((s.panel == "optimizer").sum()),
        "optimizer_below_one": int(((s.panel == "optimizer") & (s.dlr_ratio_mean < 1.0)).sum()),
        "default_reference": row_for("reference", 0.20, 0.05),
        "default_horizon": row_for("horizon", 5.0, None),
        "default_optimizer": row_for("optimizer", 0.08, 0.05),
    }
    (args.tables / "e4_dense_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
