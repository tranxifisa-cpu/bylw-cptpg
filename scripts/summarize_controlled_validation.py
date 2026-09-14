"""Reconstruct controlled results and draw evidence-led Matplotlib figures.

Figure contract: reference paths isolate history; estimator panels compare MSE
at expected cost; tracking panels distinguish residuals, smoothing and distance.
Export PDF/PNG without fabricated data, fitted convergence curves or hidden seeds.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"capped_multilevel": "#365F91", "plugin": "#B16C42", "oracle": "#3E806C",
          "frozen": "#747474", "multilevel_increasing": "#997238"}
LABELS = {"capped_multilevel": "Capped multilevel", "plugin": "Plug-in",
          "oracle": "Population gradient", "frozen": "Frozen parameter",
          "multilevel_increasing": "Increasing precision"}
SCENARIOS = {"fixed": "Fixed (p = 0.7)", "fixed_loss": "Fixed (p = 0.3)",
             "slow": "Slow variation", "switching": "Repeated switches"}


def assert_close(actual, expected):
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-13)


def check_error_summary(values, truth, row):
    values = np.asarray(values)
    squared_errors = (values - truth) ** 2
    assert_close(values.mean(), row["mean"])
    assert_close(values.mean() - truth, row["bias"])
    assert_close(values.std(ddof=1) / np.sqrt(len(values)), row["mean_se"])
    assert_close(values.var(ddof=1), row["variance"])
    assert_close(squared_errors.mean(), row["mse"])
    assert_close(squared_errors.std(ddof=1) / np.sqrt(len(values)), row["mse_se"])


def required_artifacts(suite):
    groups = {
        "estimator": ["estimator_replications.csv", "estimator_summary.csv",
                      "allocation_selection.csv", "bias_recheck.csv", "bias_recheck_summary.csv"],
        "tracking": ["tracking.csv", "tracking_summary.csv", "tracking_config.json", "target_grid.csv"],
        "path": ["reference_paths.csv"],
    }
    if suite == "all":
        return [name for files in groups.values() for name in files]
    return groups[suite]


def verify(directory):
    config = json.loads((directory / "config.json").read_text())
    completion = json.loads((directory / "completion.json").read_text())
    assert completion["status"] == "complete" and completion["suite"] == config["suite"]
    missing = [name for name in required_artifacts(config["suite"]) if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete experiment artifacts: {', '.join(missing)}")
    root = Path(__file__).resolve().parents[1]
    source_current = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() == digest
                      for p, digest in config["source_hashes"].items()}
    checks = dict(source_hashes_match=all(source_current.values()), source_details=source_current,
                  required_artifacts_present=True,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (directory / "estimator_replications.csv").exists():
        raw = pd.read_csv(directory / "estimator_replications.csv")
        summaries = pd.read_csv(directory / "estimator_summary.csv")
        keys = ["model", "method", "nominal_budget"]
        expected = {(model, method, budget) for model in ("pure_gain", "mixed_gain_loss")
                    for method in ("plugin", "capped_multilevel") for budget in config["budgets"]}
        assert set(summaries[keys].itertuples(index=False, name=None)) == expected
        assert not summaries.duplicated(keys).any()
        assert set(raw[keys].itertuples(index=False, name=None)) == expected
        assert len(raw) == len(expected) * config["repetitions"]
        levels = np.arange(config["cap"] + 1)
        probabilities = 2.0 ** (-1.5 * levels)
        cost_per_repeat = np.dot(probabilities / probabilities.sum(), 1 + 2 * 2**levels)
        tuning = pd.read_csv(directory / "allocation_selection.csv")
        assert len(tuning) == 2 * len(config["budgets"]) * 3
        for _, row in summaries.iterrows():
            records = raw[(raw.model == row.model) & (raw.method == row.method)
                          & (raw.nominal_budget == row.nominal_budget)]
            values = records.estimate.to_numpy()
            assert len(values) == row.repetitions == config["repetitions"]
            assert sorted(records.repetition) == list(range(len(values)))
            check_error_summary(values, row.true_gradient, row)
            assert_close(records.raw_trajectories.mean(), row.average_raw_cost)
            if row.method == "plugin":
                candidates = tuning[(tuning.model == row.model) & (tuning.budget == row.nominal_budget)]
                chosen = candidates.sort_values(["validation_mse", "n", "m"]).iloc[0]
                assert row.inner_n == chosen.n and row.outer_m == chosen.m
                assert row.inner_n + row.outer_m == row.nominal_budget
                assert_close(records.raw_trajectories, row.nominal_budget)
                assert_close(row.expected_raw_cost, row.nominal_budget)
            else:
                assert row.cap == config["cap"]
                assert_close(row.expected_raw_cost, row.multilevel_M * cost_per_repeat)
        checks.update(estimator_rows=len(raw), estimator_cells=len(summaries),
                      estimator_summaries_reconstructed=True, estimator_error_bars_reconstructed=True)
    if (directory / "bias_recheck.csv").exists():
        raw = pd.read_csv(directory / "bias_recheck.csv")
        summaries = pd.read_csv(directory / "bias_recheck_summary.csv")
        assert len(summaries) == 1
        summary = summaries.iloc[0]
        assert summary.n == 128
        assert len(raw) == summary.repetitions == config["bias_repetitions"]
        assert sorted(raw.repetition) == list(range(len(raw))) and (raw.n == 128).all()
        check_error_summary(raw.T_fine, summary.expected_T - summary.analytic_bias, summary)
        assert_close((raw.T_fine.mean() - summary.expected_T) / summary.mean_se,
                     summary.discrepancy_in_reported_se)
        values = raw.T_fine.to_numpy()
        top = values[np.argsort(np.abs(values))[-max(1, len(values) // 100):]].sum() / len(values)
        assert_close(top, summary.top_one_percent_absolute_contribution)
        checks.update(bias_rows=len(raw), bias_summary_reconstructed=True)
    if (directory / "tracking.csv").exists():
        data = pd.read_csv(directory / "tracking.csv")
        tc = json.loads((directory / "tracking_config.json").read_text())
        assert tc["gamma_L"] <= tc["a0"]
        assert tc["window"] == config["window"] and tc["rho"] == config["rho"]
        assert_close(tc["gamma_L"], tc["gamma"] * tc["smoothness_bound"])
        expected = set()
        for scenario in config["scenarios"]:
            methods = ["capped_multilevel", "plugin", "oracle", "frozen"]
            if config["increasing_precision"] and scenario != "switching":
                methods.append("multilevel_increasing")
            for method in methods:
                seeds = config["seeds"][:1] if method in ("oracle", "frozen") else config["seeds"]
                expected.update((scenario, method, seed) for seed in seeds)
        keys = ["scenario", "method", "seed"]
        assert set(data[keys].itertuples(index=False, name=None)) == expected
        assert len(data) == len(expected) * config["episodes"]
        for _, rows in data.groupby(["scenario", "method", "seed"], sort=False):
            rows = rows.sort_values("k")
            assert rows.k.to_list() == list(range(1, config["episodes"] + 1))
            true_g = rows.true_gradient.to_numpy()
            theta = rows.theta.to_numpy()
            true_next = np.clip(theta + tc["gamma"] * true_g / np.maximum(np.abs(true_g), tc["a0"]), -2, 2)
            q = (true_next - theta) / tc["gamma"]
            assert_close(q, rows.Q)
            direction = rows.estimated_gradient.to_numpy()
            actual_next = np.clip(theta + tc["gamma"] * direction / np.maximum(np.abs(direction), tc["a0"]), -2, 2)
            assert_close(actual_next, rows.next_theta)
            assert_close(theta[1:], rows.next_theta.to_numpy()[:-1])
            assert_close(theta[0], 0.0)
            distances = [min(abs(np.asarray(json.loads(points)) - value))
                         for points, value in zip(rows.stationary_points, theta)]
            assert_close(distances, rows.stationary_distance)
            if rows.method.iloc[0] == "frozen":
                assert_close(direction, 0.0)
                assert rows.squared_gradient_error.isna().all()
            else:
                assert_close((direction - true_g)**2, rows.squared_gradient_error)
            if rows.method.iloc[0] == "oracle":
                assert_close(direction, true_g)
            smooth = []
            for k in range(len(q)):
                index = np.arange(max(0, k - tc["window"] + 1), k + 1)
                weights = tc["rho"] ** (k - index)
                smooth.append(float(np.dot(weights, q[index]) / weights.sum()))
            cumulative = np.cumsum(np.square(smooth))
            assert_close(smooth, rows.smoothed_Q)
            assert_close(cumulative, rows.dynamic_local_regret)
            assert_close(cumulative / rows.k.to_numpy(), rows.average_dynamic_local_regret)
            assert_close(np.cumsum(q**2) / rows.k.to_numpy(), rows.average_squared_Q)
        summary = pd.read_csv(directory / "tracking_summary.csv")
        assert set(summary[keys].itertuples(index=False, name=None)) == expected
        assert not summary.duplicated(keys).any()
        for _, row in summary.iterrows():
            records = data[(data.scenario == row.scenario) & (data.method == row.method)
                           & (data.seed == row.seed)].sort_values("k")
            assert len(records) == row.episodes
            assert_close(records.raw_trajectories.sum(), row.raw_trajectories)
            assert_close(records.iloc[-1].average_squared_Q, row.average_squared_Q)
            assert_close(records.iloc[-1].average_dynamic_local_regret, row.average_dynamic_local_regret)
            assert_close(records.iloc[-1].dynamic_local_regret, row.dynamic_local_regret)
            assert_close(records.iloc[-1].next_theta, row.final_theta)
            assert_close(records.stationary_distance.mean(), row.mean_stationary_distance)
            assert_close(records.oracle_order_difference.max(), row.maximum_oracle_order_difference)
        checks.update(tracking_rows=len(data), tracking_runs=len(summary),
                      tracking_update_and_dlr_reconstructed=True, tracking_summaries_reconstructed=True,
                      distances_reconstructed_from_recorded_sets=True)
    if (directory / "reference_paths.csv").exists():
        data = pd.read_csv(directory / "reference_paths.csv")
        modes = {"asymmetric": (.4, .1), "symmetric": (.25, .25), "static": (0., 0.)}
        histories = {"up_then_down": [1., 1.4, 1.2], "down_then_up": [1., .8, 1.2]}
        assert len(data) == 6 * 33
        assert set(data[["mode", "path"]].itertuples(index=False, name=None)) == {
            (mode, path) for mode in modes for path in histories}
        for (mode, path), rows in data.groupby(["mode", "path"]):
            rows = rows.sort_values("day")
            assert rows.day.to_list() == list(range(33))
            assert_close(rows.wealth, histories[path] + [1.2] * 30)
            reference = 1.0
            gain, loss = modes[mode]
            for row in rows.itertuples():
                reference += (gain if row.wealth >= reference else loss) * (row.wealth - reference)
                assert_close(reference, row.reference)
            next_theta = np.clip(.05 * rows.prospective_gradient /
                                 np.maximum(np.abs(rows.prospective_gradient), 1.), -2., 2.)
            assert_close(1 / (1 + np.exp(-next_theta)), rows.next_mean_weight)
        checks.update(reference_rows=len(data), reference_recursions_reconstructed=True)
    (directory / "numerical_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figures(directory):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "legend.frameon": False, "lines.linewidth": 1.15})
    if (directory / "estimator_summary.csv").exists():
        frame = pd.read_csv(directory / "estimator_summary.csv")
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
        for ax, (name, part) in zip(axes, frame.groupby("model", sort=False)):
            for method, rows in part.groupby("method", sort=False):
                rows = rows.sort_values("expected_raw_cost")
                ax.errorbar(rows.expected_raw_cost, rows.mse, yerr=2 * rows.mse_se,
                            color=COLORS[method], marker="o", markersize=3,
                            capsize=2, label=LABELS[method])
            ax.set(xscale="log", yscale="log", xlabel="Expected trajectory calls", ylabel="Gradient MSE",
                   title=name.replace("_", " ").capitalize())
            ax.legend(fontsize=6)
        save_figure(fig, directory, "estimator_precision_cost")
    if (directory / "reference_paths.csv").exists():
        data = pd.read_csv(directory / "reference_paths.csv")
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
        for mode, color in zip(["asymmetric", "symmetric", "static"], ["#365F91", "#B16C42", "#747474"]):
            part = data[data["mode"] == mode]
            for path, ls in [("up_then_down", "-"), ("down_then_up", "--")]:
                rows = part[part.path == path]
                axes[0].plot(rows.day, rows.reference, linestyle=ls, color=color,
                             label=mode.capitalize() if ls == "-" else None)
            a = part[part.path == "up_then_down"].set_index("day")
            b = part[part.path == "down_then_up"].set_index("day")
            axes[1].plot(a.index[2:] - 2, np.abs(a.prospective_gradient - b.prospective_gradient).iloc[2:],
                         color=color, label=mode.capitalize())
        axes[0].axvline(2, color="#999999", linewidth=.6, linestyle=":")
        axes[0].set(xlabel="Day (shared wealth after day 2)", ylabel="Reference wealth")
        axes[1].set(xlabel="Days after matched terminal wealth", ylabel="Absolute gradient difference")
        axes[0].legend(fontsize=6)
        save_figure(fig, directory, "reference_history")
    if (directory / "tracking.csv").exists():
        data = pd.read_csv(directory / "tracking.csv")
        scenarios = list(data.scenario.unique())
        fields = [("average_squared_Q", "Mean squared residual"),
                  ("dynamic_local_regret", "Dynamic local regret"),
                  ("average_dynamic_local_regret", "Average dynamic local regret"),
                  ("stationary_distance", "Distance to numerical stationary set")]
        fig, axes = plt.subplots(len(scenarios), 4, figsize=(9.2, 2.0 * len(scenarios)), squeeze=False)
        aggregate = []
        for row, scenario in enumerate(scenarios):
            part = data[data.scenario == scenario]
            for method, records in part.groupby("method", sort=False):
                for col, (field, title) in enumerate(fields):
                    stats = records.groupby("k")[field].agg(["mean", "median", "count"])
                    quantiles = records.groupby("k")[field].quantile([.25, .75]).unstack()
                    ax = axes[row, col]
                    ax.plot(stats.index, stats["median"], color=COLORS[method], label=LABELS[method])
                    if stats["count"].max() > 1:
                        ax.fill_between(stats.index, quantiles[.25], quantiles[.75], color=COLORS[method], alpha=.14, linewidth=0)
                        ax.plot(stats.index, stats["mean"], color=COLORS[method], linestyle=":", linewidth=.65)
                    for k in stats.index:
                        aggregate.append(dict(scenario=scenario, method=method, metric=field, k=k,
                                              mean=stats.loc[k, "mean"], median=stats.loc[k, "median"],
                                              q25=quantiles.loc[k, .25], q75=quantiles.loc[k, .75], seeds=stats.loc[k, "count"]))
                    if scenario == "switching" and method == part.method.iloc[0]:
                        for start in range(100, int(records.k.max()), 200):
                            ax.axvspan(start + .5, min(start + 100.5, records.k.max()), color="#B16C42", alpha=.07, linewidth=0)
                    if row == 0:
                        ax.set_title(title, fontsize=7)
                    if col == 0:
                        ax.set_ylabel(SCENARIOS[scenario])
                    ax.set_xlabel("Episode")
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7)
        fig.tight_layout(rect=(0, .04, 1, 1), h_pad=1.8, w_pad=1.2)
        save_figure(fig, directory, "tracking_residuals")
        write_aggregate = pd.DataFrame(aggregate)
        write_aggregate.to_csv(directory / "tracking_aggregate.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    checks = verify(args.directory)
    figures(args.directory)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
