from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from .utils import ensure_dir

sns.set_theme(style="whitegrid")


METHOD_ORDER = [
    "dynamic_cpt_pg",
    "static_cpt_pg",
    "dynamic_cpt_pg_frozen_pref",
    "static_ref_dynamic_pref_cpt_pg",
    "expected_return_pg",
    "exponential_utility_pg",
]

PG_METHOD_ORDER = [
    "dynamic_cpt_pg",
    "static_cpt_pg",
    "dynamic_cpt_pg_frozen_pref",
    "static_ref_dynamic_pref_cpt_pg",
    "expected_return_pg",
    "exponential_utility_pg",
]

STYLE_TILT_CODE = {
    "momentum": 0,
    "value": 1,
    "quality": 2,
    "low_vol": 3,
    "balanced": 4,
}


def generate_plots(trace: pd.DataFrame, plot_dir: Path) -> list[Path]:
    plot_dir = ensure_dir(plot_dir)
    if trace.empty:
        return []
    paths = [
        _plot_wealth_and_cpt(trace, plot_dir / "wealth_cpt.png"),
        _plot_reference_vs_signal(trace, plot_dir / "reference_vs_signal.png"),
        _plot_gradient_norm(trace, plot_dir / "gradient_norm.png"),
    ]
    paths.extend(_plot_squared_gradient_by_method(trace, plot_dir))
    preference_path = _plot_preference_heatmap_and_turnover(trace, plot_dir / "preference_turnover.png")
    if preference_path is not None:
        paths.append(preference_path)
    return paths


def _mean_by_method(trace: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = (
        trace.groupby(["trade_date", "method"], as_index=False)[columns]
        .mean()
        .sort_values(["trade_date", "method"])
    )
    return _add_trade_date_axis(data)


def _add_trade_date_axis(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["trade_date_axis"] = pd.to_datetime(data["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    return data


def _format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)


def _format_percent_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))


def _plot_wealth_and_cpt(trace: pd.DataFrame, output_path: Path) -> Path:
    data = _mean_by_method(trace, ["wealth"])
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=data, x="trade_date_axis", y="wealth", hue="method", hue_order=METHOD_ORDER, ax=ax)
    ax.set_title("Cumulative Wealth")
    ax.set_ylabel("Wealth")
    _format_date_axis(ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_reference_vs_signal(trace: pd.DataFrame, output_path: Path) -> Path:
    data = trace[trace["method"].isin(PG_METHOD_ORDER)].copy()
    signal_column = "investment_return_rate" if "investment_return_rate" in data.columns else "day_return_rate"
    data = data.groupby(["trade_date", "method"], as_index=False)[["reference_point", signal_column]].mean()
    data = _add_trade_date_axis(data)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    sns.lineplot(data=data, x="trade_date_axis", y="reference_point", hue="method", hue_order=PG_METHOD_ORDER, ax=axes[0])
    axes[0].set_title("Reference Return Rate")
    _format_percent_axis(axes[0])
    _format_date_axis(axes[0])
    sns.lineplot(data=data, x="trade_date_axis", y=signal_column, hue="method", hue_order=PG_METHOD_ORDER, ax=axes[1], legend=False)
    axes[1].set_title("Observed Investment Net Return Rate")
    _format_percent_axis(axes[1])
    _format_date_axis(axes[1])
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_gradient_norm(trace: pd.DataFrame, output_path: Path) -> Path:
    data = trace[trace["method"].isin(PG_METHOD_ORDER)].copy()
    metric_columns = ["gradient_norm"]
    if "gradient_bootstrap_error_norm" in data.columns:
        metric_columns.append("gradient_bootstrap_error_norm")
    data = data.groupby(["trade_date", "method"], as_index=False)[metric_columns].mean()
    data = _add_trade_date_axis(data)
    has_bootstrap_error = "gradient_bootstrap_error_norm" in data.columns and data["gradient_bootstrap_error_norm"].notna().any()
    if has_bootstrap_error:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        sns.lineplot(data=data, x="trade_date_axis", y="gradient_norm", hue="method", hue_order=PG_METHOD_ORDER, ax=axes[0])
        axes[0].set_title("Gradient Norm")
        axes[0].set_ylabel("Gradient Norm")
        _format_date_axis(axes[0])
        sns.lineplot(
            data=data,
            x="trade_date_axis",
            y="gradient_bootstrap_error_norm",
            hue="method",
            hue_order=PG_METHOD_ORDER,
            ax=axes[1],
            legend=False,
        )
        axes[1].set_title("Gradient Bootstrap Error Norm")
        axes[1].set_ylabel("Bootstrap Error Norm")
        _format_date_axis(axes[1])
    else:
        fig, ax = plt.subplots(figsize=(13, 5))
        sns.lineplot(data=data, x="trade_date_axis", y="gradient_norm", hue="method", hue_order=PG_METHOD_ORDER, ax=ax)
        ax.set_title("Gradient Norm")
        _format_date_axis(ax)
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
    data = (
        data.groupby(["trade_date", "method"], as_index=False)["gradient_norm"]
        .mean()
        .sort_values(["method", "trade_date"])
    )
    data = _add_trade_date_axis(data)
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
            x="trade_date_axis",
            y="cumulative_squared_gradient_norm",
            ax=axes[0],
        )
        axes[0].set_title(f"Cumulative Squared Gradient Norm ({method})")
        axes[0].set_ylabel("Cumulative Sum")
        sns.lineplot(
            data=method_data,
            x="trade_date_axis",
            y="average_squared_gradient_norm",
            ax=axes[1],
        )
        axes[1].set_title(f"Average Squared Gradient Norm ({method})")
        axes[1].set_ylabel("Running Average")
        _format_date_axis(axes[0])
        _format_date_axis(axes[1])
        plt.tight_layout()
        output_path = plot_dir / f"squared_gradient_{method}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(output_path)
    return paths


def _plot_preference_heatmap_and_turnover(trace: pd.DataFrame, output_path: Path) -> Path | None:
    dynamic_rows = trace[trace["method"] == "dynamic_cpt_pg"]
    if dynamic_rows.empty:
        return None
    seed = int(dynamic_rows["seed"].min())
    pref_data = dynamic_rows[dynamic_rows["seed"] == seed].copy()
    if pref_data.empty:
        return None
    pref_data["style_tilt_code"] = pref_data["style_tilt"].map(STYLE_TILT_CODE).fillna(-1)
    pref_columns = [
        "risk_budget",
        "max_single_weight",
        "turnover_cap",
        "diversification_target",
        "style_tilt_code",
    ]
    pref_data["trade_date_label"] = pd.to_datetime(
        pref_data["trade_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    heatmap = pref_data[["trade_date_label", *pref_columns]].set_index("trade_date_label").T
    turnover = trace.groupby(["trade_date", "method"], as_index=False)["turnover"].mean()
    turnover = _add_trade_date_axis(turnover)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1.2, 1.0]})
    sns.heatmap(heatmap, cmap="YlGnBu", ax=axes[0])
    axes[0].set_title(f"Preference Heatmap (dynamic_cpt_pg, seed={seed})")
    sns.lineplot(data=turnover, x="trade_date_axis", y="turnover", hue="method", hue_order=METHOD_ORDER, ax=axes[1])
    axes[1].set_title("Average Turnover")
    _format_percent_axis(axes[1])
    _format_date_axis(axes[1])
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
