"""Publication plots from saved measurements, never from prescribed rankings."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import norm


COLORS = dict(dynamic="#0F4D92", symmetric="#42949E", static="#B64342",
              expected="#767676", exponential="#B58A28", equal_weight="#272727",
              plugin="#E9A6A1", frozen="#AADCA9", multilevel="#0F4D92",
              multilevel_outer="#42949E", split_base="#B58A28")
LABELS = dict(dynamic="DRCPT-PG", symmetric="Symmetric CPT-PG", static="Static CPT-PG",
              expected="Expected wealth PG", exponential="Exponential utility PG",
              equal_weight="Equal weight", plugin="Plug-in", frozen="Frozen policy",
              multilevel="Capped multilevel", multilevel_outer="Multiple outer paths",
              split_base="Separate base term")
REGIMES = dict(uptrend_momentum=("Uptrend", "#DDF3DE"),
               oscillation_reversal=("Reversal", "#CFCECE"),
               negative_shock=("Shock", "#F6CFCB"),
               style_switch_lowvol_quality=("Style switch", "#DCE8F4"))


def style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in ["Arial", "Helvetica", "DejaVu Sans"] if name in available)
    plt.rcParams.update({"font.family": family,
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1, "legend.frameon": False, "svg.fonttype": "none",
        "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def letters(axes):
    for letter, ax in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", np.asarray(axes).ravel()):
        ax.text(-.12, 1.04, letter, transform=ax.transAxes, weight="bold", fontsize=13)


def bands(ax, frame):
    unique = frame[["episode", "regime"]].drop_duplicates().sort_values("episode")
    if unique.episode.duplicated().any():
        raise ValueError("Conflicting regime labels within an episode")
    rows = list(unique.itertuples(index=False))
    if not rows:
        return
    start, current = rows[0].episode, rows[0].regime
    for episode, regime in [(r.episode, r.regime) for r in rows[1:]] + [(rows[-1].episode + 1, None)]:
        if regime != current:
            if current in REGIMES:
                ax.axvspan(start - .5, episode - .5, color=REGIMES[current][1], alpha=.38, zorder=0)
            start, current = episode, regime


def curves(ax, frame, column, ylabel, shading=True):
    if shading:
        bands(ax, frame)
    for method, data in frame.groupby("method", sort=False):
        data = data.dropna(subset=[column])
        if data.empty:
            continue
        grouped = data.groupby("episode")[column]
        median = grouped.median()
        ax.plot(median.index, median, color=COLORS[method], lw=1.8, label=LABELS[method])
        ax.fill_between(median.index, grouped.quantile(.25), grouped.quantile(.75),
                        color=COLORS[method], alpha=.16, linewidth=0)
    ax.set(xlabel="Episode", ylabel=ylabel)


def plot_path(output):
    output = Path(output)
    paths = pd.read_csv(output / "paths.csv")
    heat = pd.read_csv(output / "path_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for name, data in paths.groupby("path", sort=False):
        color = "#0F4D92" if name == "gain_first" else "#B64342"
        axes[0].plot(data.time, data.wealth, color=color, marker="o", label=f"{name}: wealth")
        axes[0].plot(data.time, data.reference, color=color, ls="--", label=f"{name}: reference")
    axes[0].set(xlabel="Observation", ylabel="Normalized level")
    axes[0].legend(fontsize=8)
    table = heat.pivot(index="eta_loss", columns="eta_gain", values="reference_gap")
    mesh = axes[1].pcolormesh(table.columns, table.index, table, shading="nearest", cmap="RdBu_r")
    axes[1].set(xlabel="Gain adaptation", ylabel="Loss adaptation")
    fig.colorbar(mesh, ax=axes[1], label="Terminal reference difference")
    letters(axes)
    save(fig, output, "experiment1_path_dependence")


def plot_estimator(output):
    output = Path(output)
    if (output / "diagnostic_summary.csv").exists():
        plot_estimator_diagnostics(output)
        return
    bias = pd.read_csv(output / "bias_summary.csv")
    variance = pd.read_csv(output / "variance_summary.csv")
    raw = pd.read_csv(output / "variance_raw.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    ax = axes[0, 0]
    for method, data in bias.groupby("estimator", sort=False):
        ax.errorbar(data.n, data.bias_coordinate, yerr=1.96 * data.bias_se,
                    marker="o", color=COLORS[method], label=LABELS[method], capsize=3)
    ax.axhline(0, color="#767676", lw=.8)
    ax.set(xscale="log", xlabel="Maximum inner sample size", ylabel="Signed coordinate bias (95% MC interval)")
    ax.legend(fontsize=8)
    d = variance.dimension.max()
    selected = variance[variance.dimension == d]
    ax = axes[0, 1]
    ax.loglog(selected.M, selected.variance_trace, "o-", color=COLORS["multilevel"], label="Measured variance")
    anchor = selected.iloc[0]
    ax.loglog(selected.M, anchor.variance_trace * anchor.M / selected.M, "--", color="#767676", label="1/M slope guide")
    ax.set(xlabel="Independent multilevel repeats M", ylabel="Trace of empirical covariance")
    ax.legend(fontsize=8)
    sample = raw[(raw.dimension == d) & (raw.M == raw.M.max())].g0.to_numpy()
    sd = sample.std(ddof=1)
    if sd <= 0:
        raise ValueError("Degenerate estimator distribution; cannot make QQ diagnostic")
    z = np.sort((sample - sample.mean()) / sd)
    theoretical = norm.ppf((np.arange(len(z)) + .5) / len(z))
    axes[1, 0].scatter(theoretical, z, s=9, color=COLORS["multilevel"])
    axes[1, 0].plot(theoretical, theoretical, color="#767676", lw=1)
    axes[1, 0].set(xlabel="Normal quantiles", ylabel="Standardized capped-estimator quantiles")
    for dimension, data in variance.groupby("dimension"):
        axes[1, 1].loglog(data.cost, data.mse, "o-", label=f"d={dimension}")
    axes[1, 1].set(xlabel="Measured mean trajectory cost", ylabel="MSE to independent numerical reference")
    axes[1, 1].legend(fontsize=8)
    letters(axes)
    save(fig, output, "experiment2_estimator_four_panels")


def plot_estimator_diagnostics(output):
    table = pd.read_csv(output / "diagnostic_summary.csv")
    levels = pd.read_csv(output / "level_summary.csv")
    for day, data in table.groupby("day"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        for method, group in data.groupby("estimator", sort=False):
            group = group.sort_values("cost")
            for ax, column, ylabel in [(axes[0, 0], "mse", "MSE to numerical reference"),
                                       (axes[0, 1], "variance_trace", "Trace of covariance")]:
                ax.loglog(group.cost, group[column], "o-", color=COLORS[method], label=LABELS[method])
                ax.set(xlabel="Measured mean trajectory cost", ylabel=ylabel)
            axes[1, 0].errorbar(group.cost, group.bias_coordinate, yerr=1.96 * group.bias_se,
                marker="o", capsize=3, color=COLORS[method], label=LABELS[method])
        axes[1, 0].axhline(0, color="#767676", lw=.8)
        axes[1, 0].set(xscale="log", xlabel="Measured mean trajectory cost",
                       ylabel="Signed coordinate bias (95% MC interval)")
        for batch, group in levels[levels.day == day].groupby("outer_batch"):
            axes[1, 1].semilogy(group.level, group.second_moment, "o-", label=f"Outer paths = {batch}")
        axes[1, 1].set(xlabel="Level (0 includes base term)", ylabel="Level second moment")
        axes[0, 0].legend(fontsize=8)
        axes[1, 1].legend(fontsize=8)
        letters(axes)
        save(fig, output, f"experiment2_diagnostics_day{day}")


def plot_tracking(output, real=False):
    output = Path(output)
    trace = pd.read_csv(output / "episodes.csv")
    daily = pd.read_csv(output / "daily.csv")
    summary = pd.read_csv(output / "summary.csv")
    if "common_objective" in trace:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
        for ax, column, ylabel in [(axes[0, 0], "common_objective", "Common-context CPT (higher is better)"),
                (axes[0, 1], "common_gradient_cross", "Independent gradient cross-product"),
                (axes[1, 0], "common_gradient_noise", "Common-context gradient noise estimate"),
                (axes[1, 1], "wealth", "Executed normalized wealth")]:
            curves(ax, trace, column, ylabel)
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].axhline(0, color="#767676", lw=.8)
        letters(axes)
        save(fig, output, "experiment3_common_context")
    if real:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained")
        curves(axes[0, 0], trace, "wealth", "Normalized wealth", False)
        end_day = daily.groupby(["method", "seed", "episode"], sort=False).tail(1)
        curves(axes[0, 1], end_day, "drawdown", "Drawdown", False)
        curves(axes[0, 2], trace, "cash", "Cash weight", False)
        curves(axes[1, 0], end_day, "cumulative_fee", "Cumulative cost / initial capital", False)
        curves(axes[1, 1], daily.groupby(["method", "seed", "episode"], as_index=False).turnover.mean(),
               "turnover", "Mean one-sided turnover", False)
        risk_scatter(axes[1, 2], summary)
        name = "experiment5_real_quote_replay"
    else:
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.3), layout="constrained")
        curves(axes[0, 0], trace, "dynamic_local_regret", "Dynamic Local Regret (estimated)")
        curves(axes[0, 1], trace, "average_dynamic_local_regret", "Average Dynamic Local Regret (estimated)")
        curves(axes[0, 2], trace, "average_q_squared", "Average squared residual (estimated)")
        curves(axes[0, 3], trace, "wealth", "Normalized wealth")
        curves(axes[1, 0], trace, "cash", "Cash weight")
        reference_method = "dynamic" if "dynamic" in trace.method.values else trace.method.iloc[0]
        dynamic = trace[trace.method == reference_method]
        curves(axes[1, 1], dynamic, "wealth", "Normalized level")
        reference = dynamic.groupby("episode").reference.median()
        axes[1, 1].plot(reference.index, reference, "--", color="#B64342", label=f"Reference: {LABELS[reference_method]}")
        axes[1, 1].legend(fontsize=8)
        risk_scatter(axes[1, 2], summary)
        # Fixed post-switch intervals, not an invented recovery-time threshold.
        switches = trace[trace.method == trace.method.iloc[0]].drop_duplicates("episode")
        points = switches.loc[switches.regime != switches.regime.shift(), "episode"].iloc[1:]
        response = []
        for point in points:
            sub = trace[(trace.episode >= point) & (trace.episode < point + 10)].copy()
            sub["episode"] -= point - 1
            response.append(sub)
        if response:
            curves(axes[1, 3], pd.concat(response).groupby(["method", "seed", "episode"],
                   as_index=False).q_squared.mean(), "q_squared", "Post-switch residual (estimated)", False)
        axes[1, 3].set_xlabel("Episodes since switch")
        name = "experiment3_tracking_merged"
    axes[0, 0].legend(fontsize=7, ncol=2)
    if not real:
        fig.legend(handles=[Patch(facecolor=color, label=name, alpha=.5)
                            for name, color in REGIMES.values()],
                   loc="lower center", bbox_to_anchor=(.5, -.04), ncol=4, fontsize=9)
    letters(axes)
    save(fig, output, name)
    for method in trace.method.unique():
        if method == "equal_weight":
            continue
        subset = trace[trace.method == method]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.3), layout="constrained")
        curves(axes[0], subset, "dynamic_local_regret", "Dynamic Local Regret (estimated)", not real)
        curves(axes[1], subset, "average_dynamic_local_regret", "Average regret (estimated)", not real)
        curves(axes[2], subset, "average_q_squared", "Average squared residual (estimated)", not real)
        curves(axes[3], subset, "diagnostic_gradient_gap", "Independent diagnostic gradient gap", not real)
        letters(axes)
        save(fig, output, f"tracking_{method}")


def risk_scatter(ax, summary):
    for method, data in summary.groupby("method", sort=False):
        ax.scatter(data.max_drawdown.median(), data.terminal_wealth.median(),
                   s=55, color=COLORS[method], label=LABELS[method])
    ax.set(xlabel="Maximum drawdown", ylabel="Terminal wealth")


def plot_ablation(output):
    output = Path(output)
    table = pd.read_csv(output / "sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    grid = table[table.sweep == "eta"].groupby(["eta_gain", "eta_loss"]).average_dlr.median().unstack(0)
    im = axes[0].pcolormesh(grid.columns, grid.index, grid, shading="nearest", cmap="cividis_r")
    axes[0].plot(grid.columns, grid.columns, "--", color="white", lw=1)
    axes[0].set(xlabel="Gain adaptation", ylabel="Loss adaptation")
    fig.colorbar(im, ax=axes[0], label="Average estimated DLR")
    for ax, kind, x, label in [(axes[1], "horizon", "horizon", "On-policy horizon h"),
                                (axes[2], "gamma", "gamma", "Step size")]:
        grouped = table[table.sweep == kind].groupby(x).average_dlr
        median = grouped.median()
        ax.plot(median.index, median, "o-", color=COLORS["dynamic"])
        ax.fill_between(median.index, grouped.quantile(.25), grouped.quantile(.75), alpha=.2)
        ax.set(xlabel=label, ylabel="Average estimated DLR")
        if kind == "gamma":
            ax.set_xscale("log")
    letters(axes)
    save(fig, output, "experiment4_ablation_sensitivity")


def plot_experiment(output, experiment):
    style()
    if experiment == 1:
        plot_path(output)
    elif experiment == 2:
        plot_estimator(output)
    elif experiment in (3, 5):
        plot_tracking(output, real=experiment == 5)
    elif experiment == 4:
        plot_ablation(output)
    else:
        raise ValueError("Only experiments 1-5 are implemented")
