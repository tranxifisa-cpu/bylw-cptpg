from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from .utils import ensure_dir

sns.set_theme(style="whitegrid")


METHOD_ORDER = [
    "dynamic_cpt_pg",
    "symmetric_cpt_pg",
    "static_cpt_pg",
    "expected_return_pg",
    "exponential_utility_pg",
]

PG_METHOD_ORDER = [
    "dynamic_cpt_pg",
    "symmetric_cpt_pg",
    "static_cpt_pg",
    "expected_return_pg",
    "exponential_utility_pg",
]


def generate_plots(trace: pd.DataFrame, plot_dir: Path) -> list[Path]:
    plot_dir = ensure_dir(plot_dir)
    if trace.empty:
        return []
    paths = [
        _plot_wealth_and_cpt(trace, plot_dir / "wealth_cpt.png"),
        _plot_reference_vs_signal(trace, plot_dir / "reference_vs_signal.png"),
        _plot_gradient_norm(trace, plot_dir / "gradient_norm.png"),
        _plot_dynamic_local_regret(trace, plot_dir / "dynamic_local_regret.png"),
    ]
    paths.extend(_plot_squared_gradient_by_method(trace, plot_dir))
    return paths


def _mean_by_method(trace: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    trace = _prepare_plot_units(trace)
    data = (
        trace.groupby(["period_axis", "method"], as_index=False)[columns]
        .mean()
        .sort_values(["period_axis", "method"])
    )
    return data


def _prepare_plot_units(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if _uses_on_policy_episodes(data):
        group_columns = ["run_key"] if "run_key" in data.columns else [col for col in ("method", "seed") if col in data.columns]
        if not group_columns:
            group_columns = ["method"] if "method" in data.columns else []
        sort_columns = [col for col in [*group_columns, "episode_index", "trade_date"] if col in data.columns]
        if sort_columns:
            data = data.sort_values(sort_columns)
        episode_columns = [*group_columns, "episode_index"]
        data = data.groupby(episode_columns, as_index=False, sort=False).tail(1).copy()
        data["period_axis"] = pd.to_numeric(data["episode_index"], errors="coerce").astype(float)
        return data
    return _add_period_axis(data)


def _uses_on_policy_episodes(data: pd.DataFrame) -> bool:
    if "episode_index" not in data.columns:
        return False
    episode_index = pd.to_numeric(data["episode_index"], errors="coerce")
    if episode_index.fillna(0.0).max() <= 0:
        return False
    if "estimation_mode" not in data.columns:
        return True
    return data["estimation_mode"].astype(str).eq("on_policy_episode").any()


def _add_period_axis(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    group_columns = ["run_key"] if "run_key" in data.columns else [col for col in ("method", "seed") if col in data.columns]
    if not group_columns:
        data["period_axis"] = pd.Series(range(1, len(data) + 1), index=data.index, dtype=float)
        return data
    sort_columns = [col for col in [*group_columns, "trade_date"] if col in data.columns]
    if not sort_columns:
        data["period_axis"] = pd.Series(range(1, len(data) + 1), index=data.index, dtype=float)
        return data
    ordered = data.sort_values(sort_columns).copy()
    ordered["period_axis"] = ordered.groupby(group_columns).cumcount() + 1
    data["period_axis"] = ordered["period_axis"].reindex(data.index).astype(float)
    return data


def _format_period_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Period")
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=10, integer=True))


def _format_percent_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))


def _plot_wealth_and_cpt(trace: pd.DataFrame, output_path: Path) -> Path:
    data = _mean_by_method(trace, ["wealth"])
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=data, x="period_axis", y="wealth", hue="method", hue_order=METHOD_ORDER, ax=ax)
    ax.set_title("Cumulative Wealth")
    ax.set_ylabel("Wealth")
    _format_period_axis(ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_reference_vs_signal(trace: pd.DataFrame, output_path: Path) -> Path:
    data = trace[trace["method"].isin(PG_METHOD_ORDER)].copy()
    signal_column = "investment_return_rate" if "investment_return_rate" in data.columns else "day_return_rate"
    data = _prepare_plot_units(data)
    data = data.groupby(["period_axis", "method"], as_index=False)[["reference_point", signal_column]].mean()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    sns.lineplot(data=data, x="period_axis", y="reference_point", hue="method", hue_order=PG_METHOD_ORDER, ax=axes[0])
    axes[0].set_title("Reference Return Rate")
    _format_percent_axis(axes[0])
    _format_period_axis(axes[0])
    sns.lineplot(data=data, x="period_axis", y=signal_column, hue="method", hue_order=PG_METHOD_ORDER, ax=axes[1], legend=False)
    axes[1].set_title("Observed Investment Net Return Rate")
    _format_percent_axis(axes[1])
    _format_period_axis(axes[1])
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_gradient_norm(trace: pd.DataFrame, output_path: Path) -> Path:
    data = trace[trace["method"].isin(PG_METHOD_ORDER)].copy()
    metric_columns = ["gradient_norm"]
    if "gradient_bootstrap_error_norm" in data.columns:
        metric_columns.append("gradient_bootstrap_error_norm")
    data = _prepare_plot_units(data)
    data = data.groupby(["period_axis", "method"], as_index=False)[metric_columns].mean()
    has_bootstrap_error = "gradient_bootstrap_error_norm" in data.columns and data["gradient_bootstrap_error_norm"].notna().any()
    if has_bootstrap_error:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        sns.lineplot(data=data, x="period_axis", y="gradient_norm", hue="method", hue_order=PG_METHOD_ORDER, ax=axes[0])
        axes[0].set_title("Gradient Norm")
        axes[0].set_ylabel("Gradient Norm")
        _format_period_axis(axes[0])
        sns.lineplot(
            data=data,
            x="period_axis",
            y="gradient_bootstrap_error_norm",
            hue="method",
            hue_order=PG_METHOD_ORDER,
            ax=axes[1],
            legend=False,
        )
        axes[1].set_title("Gradient Bootstrap Error Norm")
        axes[1].set_ylabel("Bootstrap Error Norm")
        _format_period_axis(axes[1])
    else:
        fig, ax = plt.subplots(figsize=(13, 5))
        sns.lineplot(data=data, x="period_axis", y="gradient_norm", hue="method", hue_order=PG_METHOD_ORDER, ax=ax)
        ax.set_title("Gradient Norm")
        _format_period_axis(ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_dynamic_local_regret(trace: pd.DataFrame, output_path: Path) -> Path:
    data = trace[trace["method"].isin(PG_METHOD_ORDER)].copy()
    if data.empty or "dynamic_local_regret" not in data.columns:
        return output_path
    metric_columns = [
        "dynamic_local_regret_term",
        "dynamic_local_regret",
        "average_dynamic_local_regret",
    ]
    data = _prepare_plot_units(data)
    data = data.groupby(["period_axis", "method"], as_index=False)[metric_columns].mean()
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    sns.lineplot(
        data=data,
        x="period_axis",
        y="dynamic_local_regret_term",
        hue="method",
        hue_order=PG_METHOD_ORDER,
        ax=axes[0],
    )
    axes[0].set_title("Dynamic Local Regret Term")
    axes[0].set_ylabel("Smoothed Squared Gradient")
    sns.lineplot(
        data=data,
        x="period_axis",
        y="average_dynamic_local_regret",
        hue="method",
        hue_order=PG_METHOD_ORDER,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Average Dynamic Local Regret")
    axes[1].set_ylabel("Running Average")
    sns.lineplot(
        data=data,
        x="period_axis",
        y="dynamic_local_regret",
        hue="method",
        hue_order=PG_METHOD_ORDER,
        ax=axes[2],
        legend=False,
    )
    axes[2].set_title("Dynamic Local Regret")
    axes[2].set_ylabel("Cumulative Sum")
    for ax in axes:
        _format_period_axis(ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_squared_gradient_by_method(trace: pd.DataFrame, plot_dir: Path) -> list[Path]:
    data = trace[trace["method"].isin(PG_METHOD_ORDER)].copy()
    if data.empty or "gradient_norm" not in data.columns:
        return []
    data["gradient_norm"] = pd.to_numeric(
        data["gradient_norm"],
        errors="coerce",
    )
    data = data.dropna(subset=["gradient_norm"])
    if data.empty:
        return []
    data = _prepare_plot_units(data)
    data = (
        data.groupby(["period_axis", "method"], as_index=False)["gradient_norm"]
        .mean()
        .sort_values(["method", "period_axis"])
    )
    paths: list[Path] = []
    for method in PG_METHOD_ORDER:
        method_data = data[data["method"] == method].copy()
        if method_data.empty:
            continue
        method_data["squared_gradient_norm"] = method_data["gradient_norm"] ** 2
        method_data["cumulative_squared_gradient_norm"] = method_data["squared_gradient_norm"].cumsum()
        steps = pd.Series(range(1, len(method_data) + 1), index=method_data.index, dtype=float)
        method_data["average_squared_gradient_norm"] = method_data["cumulative_squared_gradient_norm"] / steps
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        sns.lineplot(
            data=method_data,
            x="period_axis",
            y="cumulative_squared_gradient_norm",
            ax=axes[0],
        )
        axes[0].set_title(f"Cumulative Squared Gradient Norm ({method})")
        axes[0].set_ylabel("Cumulative Sum")
        sns.lineplot(
            data=method_data,
            x="period_axis",
            y="average_squared_gradient_norm",
            ax=axes[1],
        )
        axes[1].set_title(f"Average Squared Gradient Norm ({method})")
        axes[1].set_ylabel("Running Average")
        _format_period_axis(axes[0])
        _format_period_axis(axes[1])
        plt.tight_layout()
        output_path = plot_dir / f"squared_gradient_{method}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(output_path)
    return paths


