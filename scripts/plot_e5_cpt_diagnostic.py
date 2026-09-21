"""Diagnose whether formal E5 DRCPT updates improve the same-episode CPT objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mvp_cpt_pg.external_validity import replay_e5_recommendations
from mvp_cpt_pg.paper_experiments import EpisodeLaw, objective, projected_update, rng_for
from mvp_cpt_pg.paper_market import PaperMarket


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e5-run", type=Path, required=True)
    parser.add_argument("--evaluation-count", type=int, default=1024)
    return parser.parse_args()


def evaluate_update_gain(e5_run: Path, evaluation_count: int):
    if evaluation_count < 2:
        raise ValueError("evaluation-count must be at least two")
    recommendations, panel, config, _ = replay_e5_recommendations(e5_run)
    episodes = pd.read_csv(e5_run / "episodes.csv", dtype={"date": str})
    episodes = episodes[episodes.method.eq("dynamic")].copy()
    recommendations = recommendations[recommendations.method.eq("dynamic")].copy()
    market = PaperMarket(panel, semi=False)
    theta_columns = [f"theta_{index}" for index in range(config.dimension)]
    gradient_columns = [f"gradient_{index}" for index in range(config.dimension)]
    rows = []

    for seed, seed_episodes in episodes.groupby("seed", sort=True):
        previous = np.r_[1.0, np.zeros(len(panel.codes) - 1)]
        wealth, reference = 1.0, 1.0
        seed_recommendations = recommendations[recommendations.seed.eq(seed)]
        for row in seed_episodes.sort_values("episode").itertuples(index=False):
            theta = np.asarray([getattr(row, column) for column in theta_columns], dtype=float)
            gradient = np.asarray([getattr(row, column) for column in gradient_columns], dtype=float)
            theta_next = projected_update(theta, gradient, config)
            law = EpisodeLaw(market, int(row.day), wealth, reference, previous.copy(), config)
            before, _, _ = law.draw(theta, rng_for(int(seed), int(row.episode), 80), evaluation_count)
            after, _, _ = law.draw(theta_next, rng_for(int(seed), int(row.episode), 80), evaluation_count)
            objective_before = objective(before, law.cpt)
            objective_after = objective(after, law.cpt)
            rows.append(dict(seed=int(seed), episode=int(row.episode), date=str(row.date),
                             recorded_objective=float(row.objective),
                             objective_before=objective_before,
                             objective_after=objective_after,
                             update_gain=objective_after - objective_before))

            current = seed_recommendations[seed_recommendations.episode.eq(row.episode)]
            if current.empty:
                raise ValueError(f"Missing replayed recommendation for seed={seed}, episode={row.episode}")
            previous = np.asarray(current.sort_values("day").iloc[-1].weights, dtype=float)
            wealth, reference = float(row.wealth), float(row.reference)
    return pd.DataFrame(rows)


def plot_diagnostic(diagnostic: pd.DataFrame, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    objective_curve = diagnostic.groupby("episode").recorded_objective.agg(
        median="median", q25=lambda values: values.quantile(.25),
        q75=lambda values: values.quantile(.75)).reset_index()
    gain_curve = diagnostic.groupby("episode").update_gain.agg(
        median="median", q25=lambda values: values.quantile(.25),
        q75=lambda values: values.quantile(.75)).reset_index()
    gain_curve["rolling_median"] = gain_curve["median"].rolling(5, center=True, min_periods=1).mean()

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
        color = "#21618C"
        axes[0].fill_between(objective_curve.episode, objective_curve.q25, objective_curve.q75,
                             color=color, alpha=.18, linewidth=0)
        axes[0].plot(objective_curve.episode, objective_curve["median"], color=color, linewidth=1.6)
        axes[0].set(xlabel="Episode", ylabel="Estimated CPT objective",
                    title="Current-policy objective under a changing market")

        axes[1].fill_between(gain_curve.episode, gain_curve.q25, gain_curve.q75,
                             color=color, alpha=.14, linewidth=0)
        axes[1].plot(gain_curve.episode, gain_curve["median"], color=color,
                     linewidth=.8, alpha=.45, label="Episode median")
        axes[1].plot(gain_curve.episode, gain_curve.rolling_median, color="#B03A2E",
                     linewidth=1.8, label="5-episode smooth")
        axes[1].axhline(0, color="#333333", linewidth=.8, linestyle="--")
        positive = float((diagnostic.update_gain > 0).mean())
        axes[1].text(.98, .96, f"Positive updates: {positive:.1%}", transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=8)
        axes[1].set(xlabel="Episode", ylabel=r"$J_k(\theta_{k+1})-J_k(\theta_k)$",
                    title="Paired same-episode update gain")
        axes[1].legend(frameon=False, fontsize=8, loc="lower left")
        for axis, label in zip(axes, "AB"):
            axis.grid(alpha=.18, linewidth=.5)
            axis.set_axisbelow(True)
            axis.text(.015, .985, label, transform=axis.transAxes, ha="left", va="top",
                      color="white", fontweight="bold",
                      bbox=dict(boxstyle="round,pad=.18", facecolor="#243447", edgecolor="none"))
        for suffix in ("pdf", "svg", "png"):
            kwargs = {"dpi": 300} if suffix == "png" else {}
            fig.savefig(output / f"figure_E5_dynamic_cpt_optimization.{suffix}",
                        bbox_inches="tight", **kwargs)
        plt.close(fig)


def main():
    args = parse_args()
    run = args.e5_run.resolve()
    diagnostic = evaluate_update_gain(run, args.evaluation_count)
    diagnostic.to_csv(run / "cpt_optimization_diagnostic.csv", index=False)
    plot_diagnostic(diagnostic, run / "figures")
    seed_summary = diagnostic.groupby("seed").update_gain.mean()
    result = {
        "evaluation_count": args.evaluation_count,
        "updates": len(diagnostic),
        "mean_update_gain": float(diagnostic.update_gain.mean()),
        "median_update_gain": float(diagnostic.update_gain.median()),
        "positive_update_fraction": float((diagnostic.update_gain > 0).mean()),
        "positive_seed_mean_fraction": float((seed_summary > 0).mean()),
        "seed_mean_update_gain": {str(key): float(value) for key, value in seed_summary.items()},
    }
    (run / "cpt_optimization_diagnostic.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
