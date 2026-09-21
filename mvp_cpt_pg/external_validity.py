"""External-validity experiments for the paper's experiments 5--7.

The module deliberately lives outside the E1--E4 runner.  The theoretical
tracking experiments therefore keep their original protocol, while the real
market and behavior experiments have their own manifests and diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import pickle
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import xlrd

from .paper_experiments import (PaperConfig, action_map, executable_target,
                                next_wealth, policy, policy_features, rng_for,
                                run_online)
from .paper_market import FEATURES, PaperMarket, PaperPanel, file_hash, prepare_panel


METHOD_LABELS = {
    "dynamic": "DRCPT-PG",
    "symmetric": "Symmetric CPT-PG",
    "static": "Static CPT-PG",
    "expected": "Expected-Wealth PG",
    "exponential": "Exponential-Utility PG",
    "equal_weight": "Equal Weight",
}
E5_METHODS = ["dynamic", "symmetric", "static", "expected", "exponential",
              "equal_weight"]
PLOT_COLORS = {
    "dynamic": "#0072B2", "symmetric": "#D55E00", "static": "#009E73",
    "expected": "#CC79A7", "exponential": "#E69F00", "equal_weight": "#666666",
}
def _finite(value, default=0.0):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float(default)


def _normalise_code(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)
    return text


def _safe_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def augment_quality_feature(panel: PaperPanel, cache_root):
    """Add point-in-time announced ROE/ROA/ROIC to feature slot 7.

    The prepared panel carries a market-data proxy in that slot.  This function
    replaces it with the latest usable record whose announcement date is
    strictly before the trading date.  Missing fundamentals are neutral and
    reported rather than filled from the future.
    """
    cache_root = Path(cache_root)
    rows = []
    for payload, path in _cache_records(cache_root, "fina_indicator"):
        pkl = path.with_suffix(".pkl")
        if not pkl.exists():
            continue
        frame = pd.read_pickle(pkl)
        if frame.empty or "ts_code" not in frame:
            continue
        columns = [c for c in ("ts_code", "ann_date", "roe", "roa", "roic") if c in frame]
        if len(columns) < 3:
            continue
        frame = frame[columns].copy()
        frame["ts_code"] = frame["ts_code"].map(_normalise_code)
        frame["ann_date"] = frame["ann_date"].astype(str).str[:8]
        for col in ("roe", "roa", "roic"):
            if col not in frame:
                frame[col] = np.nan
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No usable fina_indicator cache under {cache_root}")
    fundamentals = pd.concat(rows, ignore_index=True)
    fundamentals = fundamentals[fundamentals.ts_code.isin(panel.codes[1:])]
    fundamentals["quality_value"] = fundamentals[["roe", "roa", "roic"]].mean(axis=1)
    fundamentals = (fundamentals.dropna(subset=["ann_date", "quality_value"])
                    .sort_values(["ts_code", "ann_date"])
                    .drop_duplicates(["ts_code", "ann_date"], keep="last"))

    coverage = []
    for day_index, date in enumerate(panel.dates):
        date = str(date)
        known = fundamentals[fundamentals.ann_date < date]
        latest = (known.sort_values("ann_date")
                  .drop_duplicates("ts_code", keep="last")
                  .set_index("ts_code"))
        quality = latest.reindex(panel.codes[1:])["quality_value"]
        values = quality.to_numpy(float)
        observed = np.isfinite(values)
        if observed.any():
            center = float(np.nanmedian(values[observed]))
            scale = float(np.nanstd(values[observed]))
            values = np.where(observed, (values - center) / (scale if scale > 0 else 1.0), 0.0)
            values = np.clip(values, -3.0, 3.0) / 3.0
        else:
            values = np.zeros(len(panel.codes) - 1)
        panel.features[day_index, 1:, FEATURES.index("quality_signal")] = values
        coverage.append(dict(date=date, observed_assets=int(observed.sum()),
                             total_assets=len(observed), coverage=float(observed.mean())))
    return pd.DataFrame(coverage)


def _cache_records(root, namespace):
    for path in sorted((Path(root) / namespace).glob("*.json")):
        yield _safe_json(path), path


def run_equal_weight_benchmark(market: PaperMarket, config: PaperConfig, seeds, start_day, steps):
    rows = []
    for seed in seeds:
        returns = market.execution_returns(seed)
        wealth, previous, peak, cumulative_fee = 1.0, np.r_[1.0, np.zeros(len(market.panel.codes) - 1)], 1.0, 0.0
        for day in range(start_day, start_day + steps):
            tradable = np.asarray(market.panel.tradable[day, 1:], dtype=bool)
            target = np.zeros(len(previous))
            if tradable.any():
                target[1:][tradable] = 1.0 / tradable.sum()
            else:
                target[0] = 1.0
            target = action_map(previous, target, market.panel.tradable[day], config.trade_fraction)
            old_wealth = wealth
            wealth, fee, turnover = next_wealth(wealth, previous, target, returns[day], config)
            cumulative_fee += float(fee)
            peak = max(peak, float(wealth))
            risky = target[1:]
            risky_total = float(risky.sum())
            effective_holdings = (risky_total ** 2 / float(risky @ risky)
                                  if float(risky @ risky) > 0 else 0.0)
            rows.append(dict(seed=int(seed), method="equal_weight", episode=(day - start_day) // config.horizon + 1,
                             day=day, date=str(market.panel.dates[day]), regime=market.regime(day),
                             wealth=float(wealth), reference=np.nan,
                             net_return=float(wealth / old_wealth - 1.0),
                             drawdown=float(1.0 - wealth / peak), cash=float(target[0]),
                             turnover=float(turnover), fee=float(fee), cumulative_fee=cumulative_fee,
                             effective_holdings=effective_holdings,
                             max_stock_weight=float(risky.max(initial=0.0))))
            previous = target
    return pd.DataFrame(rows)


def summarize_external(daily):
    rows = []
    for (method, seed), frame in daily.groupby(["method", "seed"], sort=False):
        returns = frame.net_return.to_numpy(float)
        volatility = returns.std(ddof=1) if len(returns) > 1 else np.nan
        terminal = float(frame.wealth.iloc[-1])
        turnover = float(frame.turnover.sum())
        rows.append(dict(method=method, method_label=METHOD_LABELS.get(method, method), seed=int(seed),
                         terminal_wealth=terminal,
                         annualized_return=terminal ** (252.0 / len(frame)) - 1.0,
                         annualized_volatility=volatility * math.sqrt(252.0) if np.isfinite(volatility) else np.nan,
                         sharpe=math.sqrt(252.0) * returns.mean() / volatility if volatility > 0 else np.nan,
                         max_drawdown=float(frame.drawdown.max()), total_turnover=turnover,
                         return_per_turnover=(terminal - 1.0) / turnover if turnover > 0 else np.nan,
                         cumulative_fee=float(frame.fee.sum()), mean_cash=float(frame.cash.mean()),
                         mean_effective_holdings=float(frame.effective_holdings.mean()),
                         mean_max_stock_weight=float(frame.max_stock_weight.mean())))
    return pd.DataFrame(rows)


def _episode_band(frame, value):
    grouped = frame.groupby("episode")[value]
    return grouped.median(), grouped.quantile(.25), grouped.quantile(.75)


def _market_stage_table(panel, start_day, steps, horizon):
    returns = np.nanmean(panel.returns[start_day:start_day + steps, 1:], axis=1)
    rows = []
    for episode in range(steps // horizon):
        block = returns[episode * horizon:(episode + 1) * horizon]
        rows.append(dict(episode=episode + 1, market_return=float(np.nanmean(block)), stage=""))
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Smooth the observed market return before assigning visual stages.  This
    # is only a figure annotation, not an input to the trading strategy.
    smoothed = table.market_return.rolling(5, center=True, min_periods=1).mean()
    shock_cut = smoothed.quantile(.10)
    high_cut = smoothed.quantile(.65)
    low_cut = smoothed.quantile(.35)
    table["stage"] = np.select(
        [smoothed <= shock_cut, smoothed >= high_cut, smoothed <= low_cut],
        ["Shock", "Uptrend", "Reversal"], default="Sideways/Recovery")
    return table


def _add_stage_bands(ax, stages, colors):
    if stages.empty:
        return
    start_episode = int(stages.episode.iloc[0])
    current = stages.stage.iloc[0]
    for index in range(1, len(stages) + 1):
        end_episode = int(stages.episode.iloc[index - 1])
        boundary = end_episode + .5
        if index == len(stages) or stages.stage.iloc[index] != current:
            ax.axvspan(start_episode - .5, boundary, color=colors[current], alpha=.16, lw=0)
            if index < len(stages):
                start_episode, current = int(stages.episode.iloc[index]), stages.stage.iloc[index]


def _panel_label(ax, label):
    ax.text(.015, .97, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=.18", facecolor="#263746", edgecolor="none"))


def plot_e5(daily, summary, output, panel=None, start_day=None, horizon=5):
    """Plot the four-panel real-market external-validity figure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = [m for m in E5_METHODS if m in set(daily.method)]
    stages = (_market_stage_table(panel, start_day, len(daily[daily.method == methods[0]].day.unique()), horizon)
              if panel is not None and start_day is not None and methods else pd.DataFrame())
    stage_colors = {"Uptrend": "#DDEFE0", "Reversal": "#E5E7F5",
                    "Shock": "#F6E5D7", "Sideways/Recovery": "#EEF2E9"}
    daily = daily.copy()
    if not stages.empty:
        daily = daily.merge(stages[["episode", "stage"]], on="episode", how="left")
    with plt.rc_context({"font.family": "DejaVu Sans",
                         "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), constrained_layout=True)
        for method in methods:
            frame = daily[daily.method == method]
            median, lower, upper = _episode_band(frame, "wealth")
            x = median.index.to_numpy()
            axes[0, 0].plot(x, median, label=METHOD_LABELS[method], color=PLOT_COLORS[method], lw=1.9)
            axes[0, 0].fill_between(x, lower.to_numpy(), upper.to_numpy(), color=PLOT_COLORS[method], alpha=.11)
            median, lower, upper = _episode_band(frame, "drawdown")
            axes[0, 1].plot(x, median, label=METHOD_LABELS[method], color=PLOT_COLORS[method], lw=1.7)
            axes[0, 1].fill_between(x, lower.to_numpy(), upper.to_numpy(), color=PLOT_COLORS[method], alpha=.10)
        if not stages.empty:
            _add_stage_bands(axes[0, 0], stages, stage_colors)
            _add_stage_bands(axes[0, 1], stages, stage_colors)
        axes[0, 0].set_title("A-share real-market wealth replay")
        axes[0, 1].set_title("Stress-period drawdown and recovery")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Normalized wealth")
        axes[0, 1].set_ylabel("Drawdown from running peak (lower is better)")
        stage_handles = [Patch(facecolor=stage_colors[name], edgecolor="none", alpha=.55, label=name)
                         for name in stage_colors]
        axes[0, 1].legend(handles=stage_handles, title="Market stage", frameon=False,
                          fontsize=7, title_fontsize=7, loc="upper right", ncol=2)

        aggregate = summary[summary.method.isin(methods)].groupby("method", sort=False).median(numeric_only=True).reset_index()
        max_turnover = max(float(aggregate.total_turnover.max()), 1e-12)
        for _, row in aggregate.iterrows():
            method = row.method
            axes[1, 0].scatter(row.max_drawdown, row.terminal_wealth,
                               color=PLOT_COLORS[method],
                               s=180 + 720 * row.total_turnover / max_turnover,
                               alpha=.84, edgecolor="white", linewidth=.7)
        axes[1, 0].set_title("Return-risk-turnover Pareto lens")
        axes[1, 0].set_xlabel("Maximum drawdown (lower is better)")
        axes[1, 0].set_ylabel("Terminal wealth")

        # The two metrics have different units, so use paired horizontal bars
        # with separate x-axes while keeping a common method order.
        mean_metrics = (summary[summary.method.isin(methods)]
                        .groupby("method", sort=False)[["total_turnover", "cumulative_fee"]]
                        .mean().reindex(methods).reset_index())
        y = np.arange(len(mean_metrics))
        ax_turnover = axes[1, 1]
        ax_cost = ax_turnover.twiny()
        colors = [PLOT_COLORS[m] for m in mean_metrics.method]
        turnover_bars = ax_turnover.barh(y - .18, mean_metrics.total_turnover,
                                         height=.30, color=colors, alpha=.90)
        cost_bars = ax_cost.barh(y + .18, mean_metrics.cumulative_fee,
                                 height=.30, color=colors, alpha=.48, hatch="//")
        ax_turnover.set_yticks(y, [METHOD_LABELS[m] for m in mean_metrics.method], fontsize=8)
        ax_turnover.invert_yaxis()
        ax_turnover.set_title("Trading activity and transaction cost")
        ax_turnover.set_xlabel("Mean total turnover")
        ax_cost.set_xlabel("Mean cumulative transaction cost")
        ax_cost.tick_params(axis="y", left=False, right=False,
                            labelleft=False, labelright=False)
        ax_cost.grid(False)
        ax_cost.patch.set_alpha(0)
        ax_cost.spines["top"].set_visible(True)
        for bar, value in zip(turnover_bars, mean_metrics.total_turnover):
            ax_turnover.text(value, bar.get_y() + bar.get_height() / 2,
                             f" {value:.1f}", va="center", fontsize=7)
        for bar, value in zip(cost_bars, mean_metrics.cumulative_fee):
            ax_cost.text(value, bar.get_y() + bar.get_height() / 2,
                         f" {value:.3f}", va="center", fontsize=7)
        for ax in axes.flat:
            ax.grid(alpha=.18, linewidth=.6)
            ax.set_axisbelow(True)
        axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
        for ax, label in zip(axes.flat, "ABCD"):
            _panel_label(ax, label)
        fig.savefig(output / "figure_E5_external_validity.pdf", bbox_inches="tight")
        fig.savefig(output / "figure_E5_external_validity.svg", bbox_inches="tight")
        fig.savefig(output / "figure_E5_external_validity.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def prepare_real_panel(cache_root, panel_path, start, end, assets, pool_seed, allow_suspension_carry):
    metadata = prepare_panel(cache_root, panel_path, start, end, assets, pool_seed, allow_suspension_carry)
    return metadata


def run_e5(cache_root, panel_path, output, start_date="20230103", test_start="20250102",
           end_date="20260529", assets=300, pool_seed=2022, seeds=(29, 147, 3141),
           horizon=5, gamma=.08, trajectory_budget=512, evaluation_n=512,
           evaluation_m=256, eta_gain=.2, eta_loss=.05, cost=.001,
           trade_fraction=.2, policy_sharpness=6.0, dirichlet_scale=10.0,
           allow_suspension_carry=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    panel_path = Path(panel_path)
    if not panel_path.exists():
        prepare_real_panel(cache_root, panel_path, start_date, end_date, assets, pool_seed,
                           allow_suspension_carry)
    panel = PaperPanel.load(panel_path, start_date, end_date)
    if len(panel.codes) - 1 != assets:
        raise ValueError(f"Panel has {len(panel.codes) - 1} assets, expected {assets}")
    coverage = augment_quality_feature(panel, cache_root)
    market = PaperMarket(panel, semi=False)
    start_day = int(np.searchsorted(panel.dates, test_start))
    config = PaperConfig(horizon=horizon, gamma=gamma, a0=.05,
                         eta_gain=eta_gain, eta_loss=eta_loss,
                         cost=cost, trajectory_budget=trajectory_budget,
                         evaluation_n=evaluation_n, evaluation_m=evaluation_m,
                         dimension=min(10, panel.features.shape[-1]), trade_fraction=trade_fraction,
                         policy_sharpness=policy_sharpness,
                         dirichlet_scale=dirichlet_scale)
    available = len(panel.dates) - start_day
    steps = available - available % horizon
    if start_day <= horizon or steps < horizon:
        raise ValueError("Real-data test period is too short for the configured horizon")
    learned, learning_daily = run_online(market, config, list(seeds),
                                         ["dynamic", "symmetric", "static", "expected", "exponential"],
                                         steps, start_day, estimator="hybrid",
                                         average_training_outputs=True)
    benchmark_daily = run_equal_weight_benchmark(market, config, list(seeds), start_day, steps)
    daily = pd.concat([learning_daily, benchmark_daily], ignore_index=True)
    summary = summarize_external(daily)
    learned.to_csv(output / "episodes.csv", index=False)
    daily.to_csv(output / "daily.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    coverage.to_csv(output / "quality_coverage.csv", index=False)
    manifest = dict(experiment=5, status="complete", panel=str(panel_path), panel_sha256=file_hash(panel_path),
                    cache_root=str(cache_root), start_date=start_date, test_start=test_start,
                    end_date=end_date, assets=assets, pool_seed=pool_seed, seeds=list(seeds),
                    config=asdict(config), methods=E5_METHODS,
                    estimator="hybrid (centered LOO base + independent randomized correction)",
                    training_output_schedule="M_k=max(4,ceil(sqrt(k*h/5))); independent outputs averaged",
                    factor_timing="all features use information strictly before the execution day; announced fundamentals use ann_date < date",
                    trajectory_sampling="uniform pre-episode historical blocks with date-aligned factors, tradability and returns at every step",
                    return_basis="close/pre_close quote-relative; not a dividend-adjusted total-return series",
                    transaction_cost="c * ||target_t - target_{t-1}||_1, charged in next_wealth",
                    dlr_warning="DLR is not used to rank real-market methods; this is an external-validity backtest")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_e5(daily, summary, output / "figures", panel=panel, start_day=start_day, horizon=horizon)
    return summary


def load_corporate_actions(behavior_root):
    """Read ex-date corporate actions without relying on the broken headers."""
    actions = {}
    behavior_root = Path(behavior_root)
    for path in (behavior_root / "xrxd1.xls", behavior_root / "xrxd2.xls"):
        if not path.exists():
            continue
        book = xlrd.open_workbook(str(path), on_demand=True)
        sheet = book.sheet_by_index(0)
        for row in range(1, sheet.nrows):
            values = sheet.row_values(row)
            code = _normalise_code(values[3])
            if not code:
                continue
            try:
                ex_date = xlrd.xldate_as_datetime(float(values[27]), book.datemode).strftime("%Y-%m-%d")
            except (TypeError, ValueError, xlrd.XLDateError):
                continue
            dividend = _finite(values[29])
            delivery = _finite(values[32])
            issue = _finite(values[33])
            record = (ex_date, dividend, 1.0 + delivery + issue)
            # Corporate-action tables store the six-digit code without the
            # exchange suffix used by Tushare/trading records.
            for alias in (code, "SH" + code, "SZ" + code):
                actions.setdefault(alias, []).append(record)
        book.release_resources()
    for code in actions:
        actions[code] = sorted(actions[code])
    return actions


def _read_rebalancing_events(behavior_root, row_limit=None):
    required = ["status", "id", "prev_weight_adjusted", "price", "proactive",
                "stock_symbol", "target_weight", "updated_at_rebalancing_histories", "sp"]
    frames = []
    paths = sorted((Path(behavior_root) / "raw_data" / "trading_data").glob("tm_trading*.xlsx"))
    if not paths:
        raise FileNotFoundError("No raw trading workbooks under behavior/raw_data/trading_data")
    for path in paths:
        read_kwargs = {"usecols": required}
        if row_limit is not None:
            read_kwargs["nrows"] = int(row_limit)
        frame = pd.read_excel(path, **read_kwargs)
        frame = frame[frame.status.astype(str).str.lower().eq("success")].copy()
        frame["sp"] = frame.sp.map(_normalise_code)
        frame["stock_symbol"] = frame.stock_symbol.map(_normalise_code)
        frame["date"] = pd.to_datetime(frame.updated_at_rebalancing_histories, errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ("id", "prev_weight_adjusted", "price", "target_weight"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["proactive"] = frame.proactive.astype(str).str.lower().isin(("true", "1", "yes"))
        frames.append(frame.dropna(subset=["sp", "stock_symbol", "date", "price", "target_weight"]))
    events = pd.concat(frames, ignore_index=True)
    # Sort before reconstructing holdings.  The original workbooks are chunks,
    # not guaranteed chronological partitions.
    return events.sort_values(["sp", "date", "id", "stock_symbol"], kind="mergesort").reset_index(drop=True)


def _apply_actions(value, code, before_date, after_date, actions):
    value = float(value)
    for ex_date, dividend, factor in actions.get(code, ()):
        if before_date < ex_date <= after_date:
            value = (value - dividend) / max(factor, np.finfo(float).tiny)
    return value


def _processed_opportunities(behavior_root, output, max_opportunities=0, epsilon=1e-6,
                             eta_gain=.4, eta_loss=.1):
    """Build E6 rows from the existing 6purchase processing archive.

    The archive already contains point-in-time purchase-cost dictionaries after
    corporate actions, so re-reading the 7.8m-row raw workbooks is unnecessary.
    """
    archive = Path(behavior_root) / "processing_process.zip"
    if not archive.exists():
        return None
    names = []
    with zipfile.ZipFile(archive) as zf:
        names = sorted(name for name in zf.namelist()
                       if name.startswith("processing_process/6purchase/") and name.endswith(".pkl"))
        if not names:
            return None
        records = []
        for name in names:
            frame = pickle.loads(zf.read(name))
            required = {"sp", "day", "stock_symbol", "prev_weight_adjusted", "target_weight", "price",
                        "stock_pool_beforexrxdpurchase", "stock_pool_afterxrxdpurchase"}
            if not required.issubset(frame.columns):
                continue
            frame = frame.sort_values(["day", "id", "stock_symbol"], kind="mergesort")
            event_counts = frame.groupby(frame.day.astype(str)).size().to_dict()
            state = {}
            for _, row in frame.iterrows():
                code = _normalise_code(row.stock_symbol)
                date = str(row.day)[:10]
                prev_weight = _finite(row.prev_weight_adjusted)
                target_weight = _finite(row.target_weight)
                price = max(_finite(row.price), np.finfo(float).tiny)
                before = row.stock_pool_beforexrxdpurchase if isinstance(row.stock_pool_beforexrxdpurchase, dict) else {}
                after = row.stock_pool_afterxrxdpurchase if isinstance(row.stock_pool_afterxrxdpurchase, dict) else {}
                basis = _finite(before.get(code), np.nan)
                current = state.get(code)
                if prev_weight > epsilon and np.isfinite(basis):
                    if current is None:
                        current = dict(symmetric_ref=basis, asymmetric_ref=basis,
                                       purchase_date=date, last_date=date)
                        state[code] = current
                    symmetric_relative = price / max(current["symmetric_ref"], np.finfo(float).tiny) - 1.0
                    asymmetric_relative = price / max(current["asymmetric_ref"], np.finfo(float).tiny) - 1.0
                    records.append(dict(user_id=_normalise_code(row.zhuliren), sp=_normalise_code(row.sp), date=date,
                                        stock_symbol=code, y=int(target_weight < prev_weight - epsilon),
                                        price=price, prev_weight=prev_weight / 100.0,
                                        age_days=(pd.Timestamp(date) - pd.Timestamp(current["purchase_date"])).days,
                                        proactive=int(bool(row.proactive)), event_count=int(event_counts.get(str(row.day), 1)),
                                        static_relative=price / max(basis, np.finfo(float).tiny) - 1.0,
                                        symmetric_relative=symmetric_relative, asymmetric_relative=asymmetric_relative,
                                        static_gain=int(basis <= price), symmetric_gain=int(symmetric_relative >= 0),
                                        asymmetric_gain=int(asymmetric_relative >= 0)))
                    if max_opportunities and len(records) >= max_opportunities:
                        break
                if current is None:
                    current = dict(symmetric_ref=basis if np.isfinite(basis) else price,
                                   asymmetric_ref=basis if np.isfinite(basis) else price,
                                   purchase_date=date, last_date=date)
                    state[code] = current
                if target_weight > prev_weight + epsilon:
                    new_basis = _finite(after.get(code), price)
                    current["symmetric_ref"] = new_basis
                    current["asymmetric_ref"] = new_basis
                    current["purchase_date"] = date
                else:
                    gain_eta = eta_gain if price >= current["asymmetric_ref"] else eta_loss
                    symmetric_eta = .5 * (eta_gain + eta_loss)
                    current["symmetric_ref"] += symmetric_eta * (price - current["symmetric_ref"])
                    current["asymmetric_ref"] += gain_eta * (price - current["asymmetric_ref"])
                current["last_date"] = date
                if target_weight <= epsilon:
                    state.pop(code, None)
            if max_opportunities and len(records) >= max_opportunities:
                break
    if not records:
        raise ValueError("No eligible opportunities in processed 6purchase archive")
    frame = _attach_predecision_controls(pd.DataFrame(records), behavior_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    metadata = dict(rows=len(frame), source=str(archive), processed_layer="6purchase",
                    label="target weight decreases among stocks held immediately before the rebalance",
                    purchase_cost="stock_pool_beforexrxdpurchase after corporate-action adjustment",
                    max_opportunities=max_opportunities, eta_gain=eta_gain, eta_loss=eta_loss,
                    outcome="target-weight decrease among previously held stocks",
                    warning="paper-defined PGR/PLR fields are not used to compare reference mechanisms")
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def build_e6_opportunities(behavior_root, output, max_opportunities=0, epsilon=1e-6,
                           eta_gain=.4, eta_loss=.1):
    """Reconstruct held-stock reduce opportunities from target-weight records.

    A row is produced only when the stock was held immediately before the
    rebalance and its target weight decreases.  Purchase-cost references are
    updated after the label is formed, so the label cannot leak the current
    decision into its own reference.
    """
    behavior_root, output = Path(behavior_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    processed = _processed_opportunities(behavior_root, output, max_opportunities, epsilon,
                                         eta_gain, eta_loss)
    if processed is not None:
        return processed
    actions = load_corporate_actions(behavior_root)
    read_limit = None if not max_opportunities else max(10_000, int(max_opportunities) * 100)
    events = _read_rebalancing_events(behavior_root, read_limit)
    mapping = pd.read_excel(behavior_root / "tm_user_sp_all.xlsx")
    sp_to_user = dict(zip(mapping.sp.map(_normalise_code), mapping.user))
    records = []
    for sp, group in events.groupby("sp", sort=False):
        state = {}
        event_counts = group.groupby("date").size().to_dict()
        for _, row in group.iterrows():
            code, date = row.stock_symbol, row.date
            prev_weight = _finite(row.prev_weight_adjusted)
            target_weight = _finite(row.target_weight)
            price = max(_finite(row.price), np.finfo(float).tiny)
            current = state.get(code)
            if prev_weight > epsilon and current is not None:
                current["basis"] = _apply_actions(current["basis"], code, current["last_date"], date, actions)
                current["symmetric_ref"] = _apply_actions(current["symmetric_ref"], code, current["last_date"], date, actions)
                current["asymmetric_ref"] = _apply_actions(current["asymmetric_ref"], code, current["last_date"], date, actions)
                static_relative = price / max(current["basis"], np.finfo(float).tiny) - 1.0
                symmetric_relative = price / max(current["symmetric_ref"], np.finfo(float).tiny) - 1.0
                asymmetric_relative = price / max(current["asymmetric_ref"], np.finfo(float).tiny) - 1.0
                records.append(dict(user_id=sp_to_user.get(sp, sp), sp=sp, date=date,
                                    stock_symbol=code, y=int(target_weight < prev_weight - epsilon),
                                    price=price, prev_weight=prev_weight / 100.0,
                                    age_days=(pd.Timestamp(date) - pd.Timestamp(current["purchase_date"])).days,
                                    proactive=int(bool(row.proactive)), event_count=int(event_counts.get(date, 1)),
                                    static_relative=static_relative, symmetric_relative=symmetric_relative,
                                    asymmetric_relative=asymmetric_relative,
                                    static_gain=int(static_relative >= 0), symmetric_gain=int(symmetric_relative >= 0),
                                    asymmetric_gain=int(asymmetric_relative >= 0)))
                if max_opportunities and len(records) >= max_opportunities:
                    break
            if current is None:
                current = dict(basis=price, symmetric_ref=price, asymmetric_ref=price,
                               purchase_date=date, last_date=date)
                state[code] = current
            else:
                if target_weight > prev_weight + epsilon:
                    target = max(target_weight, epsilon)
                    increment = target_weight - prev_weight
                    current["basis"] = (current["basis"] * prev_weight + price * increment) / target
                    current["symmetric_ref"] = (current["symmetric_ref"] * prev_weight + price * increment) / target
                    current["asymmetric_ref"] = (current["asymmetric_ref"] * prev_weight + price * increment) / target
                    current["purchase_date"] = date
                else:
                    symmetric_eta = .5 * (eta_gain + eta_loss)
                    gain_eta = eta_gain if price >= current["asymmetric_ref"] else eta_loss
                    current["symmetric_ref"] += symmetric_eta * (price - current["symmetric_ref"])
                    current["asymmetric_ref"] += gain_eta * (price - current["asymmetric_ref"])
                current["last_date"] = date
            if target_weight <= epsilon:
                state.pop(code, None)
        if max_opportunities and len(records) >= max_opportunities:
            break
    if not records:
        raise ValueError("No held-stock opportunities were reconstructed")
    frame = _attach_predecision_controls(pd.DataFrame(records), behavior_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    metadata = dict(rows=len(frame), source_files=[str(p) for p in sorted((behavior_root / "raw_data" / "trading_data").glob("tm_trading*.xlsx"))],
                    corporate_action_files=[str(behavior_root / "xrxd1.xls"), str(behavior_root / "xrxd2.xls")],
                    label="target weight decreases among stocks held immediately before the rebalance",
                    epsilon=epsilon, max_opportunities=max_opportunities,
                    eta_gain=eta_gain, eta_loss=eta_loss,
                    outcome="target-weight decrease among previously held stocks",
                    warning="paper-defined PGR/PLR fields are not used to compare reference mechanisms")
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def _attach_predecision_controls(frame, behavior_root):
    """Attach lagged controls without importing static-reference outcomes.

    The processed regression table already contains the paper's daily market,
    holding and social aggregates.  We lag them by investor before merging, so
    a current trading decision cannot contribute to its own predictors.
    """
    source_path = Path(behavior_root) / "regression_afterholiday.pkl"
    if not source_path.exists():
        return frame
    source = pd.read_pickle(source_path).copy()
    source["user_id"] = source.zhuliren.map(_normalise_code)
    source["date"] = pd.to_datetime(source.day, errors="coerce").dt.normalize()
    source = source.sort_values(["user_id", "date"], kind="mergesort")
    source_columns = ("mktret", "transnum", "stknum", "stkprice", "num_posts",
                      "pos", "neg", "neu")
    for column in source_columns:
        source[column] = pd.to_numeric(source[column], errors="coerce")
        source[f"lag_{column}"] = source.groupby("user_id")[column].shift(1)
    lag_columns = [f"lag_{column}" for column in source_columns]
    source = source[["user_id", "date", *lag_columns]]
    source = source.drop_duplicates(["user_id", "date"])
    result = frame.copy()
    result["user_id"] = result.user_id.map(_normalise_code)
    result["date"] = pd.to_datetime(result.date, errors="coerce").dt.normalize()
    result = result.merge(source, on=["user_id", "date"], how="left", validate="many_to_one")
    return result


def _add_behavior_controls(frame, reference):
    x = frame.copy()
    x["log_age"] = np.log1p(np.maximum(x.age_days, 0))
    x["log_event_count"] = np.log1p(np.maximum(x.event_count, 0))
    x["log_prev_weight"] = np.log1p(np.maximum(x.prev_weight, 0))
    x["log_price"] = np.log1p(np.maximum(pd.to_numeric(x.price, errors="coerce"), 0))
    x["proactive"] = x.proactive.astype(float)
    controls = ["log_age", "log_event_count", "log_prev_weight", "log_price", "proactive",
                "log_prior_opportunities", "prior_reduce_rate",
                "lag_mktret", "lag_transnum", "lag_stknum", "lag_stkprice", "lag_num_posts",
                "lag_pos", "lag_neg", "lag_neu"]
    if reference == "asymmetric":
        controls += ["asymmetric_relative", "asymmetric_gain"]
    elif reference == "symmetric":
        controls += ["symmetric_relative", "symmetric_gain"]
    elif reference == "static":
        controls += ["static_relative", "static_gain"]
    elif reference == "none":
        pass
    else:
        raise ValueError(reference)
    values = x.reindex(columns=controls).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    return values.to_numpy(float), controls


def _calibration(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    design = np.c_[np.ones(len(z)), z]
    coef, *_ = np.linalg.lstsq(design, np.asarray(y, float), rcond=None)
    return float(coef[0]), float(coef[1])


def _classification_metrics(y, p):
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    result = dict(log_loss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p),
                  calibration_intercept=_calibration(p, y)[0], calibration_slope=_calibration(p, y)[1],
                  observed_rate=float(y.mean()), predicted_rate=float(p.mean()))
    result["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    result["auprc"] = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    return result


def _reference_disposition_metrics(frame, reference):
    """Measure realized reduction behavior under one reference partition."""
    if reference not in ("static", "symmetric", "asymmetric"):
        raise ValueError(reference)
    y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(float)
    gain = pd.to_numeric(frame[f"{reference}_gain"], errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(gain)
    y = y[valid]
    gain = gain[valid] >= .5
    gain_count = int(gain.sum())
    loss_count = int((~gain).sum())
    gain_rate = float(y[gain].mean()) if gain_count else np.nan
    loss_rate = float(y[~gain].mean()) if loss_count else np.nan
    gap = gain_rate - loss_rate if np.isfinite(gain_rate) and np.isfinite(loss_rate) else np.nan
    return dict(gain_reduction_rate=gain_rate, loss_reduction_rate=loss_rate,
                disposition_gap=gap, gain_opportunities=gain_count,
                loss_opportunities=loss_count)


def fit_e6_models(opportunities, output, bootstrap=200, seed=2026,
                  eta_gain=.4, eta_loss=.1, opportunities_source=None):
    frame = opportunities.copy()
    frame["date"] = pd.to_datetime(frame.date)
    frame["_user"] = frame.user_id.map(_normalise_code)
    frame = frame.sort_values(["date", "_user", "stock_symbol"], kind="mergesort").reset_index(drop=True)
    prior_count = frame.groupby("_user", sort=False).cumcount().astype(float)
    prior_sum = frame.groupby("_user", sort=False).y.transform(lambda values: values.shift(1).cumsum())
    frame["log_prior_opportunities"] = np.log1p(prior_count)
    frame["prior_reduce_rate"] = np.divide(
        prior_sum.to_numpy(float), prior_count.to_numpy(float),
        out=np.zeros(len(frame), dtype=float), where=prior_count.to_numpy(float) > 0)
    unique_dates = np.array(sorted(frame.date.dt.normalize().unique()))
    train_cut = unique_dates[int(.70 * len(unique_dates))]
    valid_cut = unique_dates[int(.85 * len(unique_dates))]
    train = frame[frame.date < train_cut]
    valid = frame[(frame.date >= train_cut) & (frame.date < valid_cut)]
    test = frame[frame.date >= valid_cut].copy()
    if train.empty or valid.empty or test.empty:
        raise ValueError("E6 chronological split is empty; inspect opportunity dates")
    rows, predictions = [], []
    for reference in ("none", "static", "symmetric", "asymmetric"):
        x_train, names = _add_behavior_controls(train, reference)
        x_valid, _ = _add_behavior_controls(valid, reference)
        x_test, _ = _add_behavior_controls(test, reference)
        scaler = StandardScaler().fit(x_train)
        y_train = train.y.to_numpy(int)
        if len(np.unique(y_train)) < 2:
            model = DummyClassifier(strategy="prior").fit(scaler.transform(x_train), y_train)
        else:
            # Keep the empirical event prevalence for probability metrics.
            # Class balancing changes the intercept and would contaminate
            # log-loss, Brier score and PGR/PLR aggregation.
            model = LogisticRegression(max_iter=400, class_weight=None, random_state=seed)
            model.fit(scaler.transform(x_train), y_train)
        # Validation is retained in the output to make the selection protocol auditable;
        # no test observation is used for fitting or threshold selection.
        p_valid = model.predict_proba(scaler.transform(x_valid))[:, 1]
        p_test = model.predict_proba(scaler.transform(x_test))[:, 1]
        valid_metrics = _classification_metrics(valid.y, p_valid)
        test_metrics = _classification_metrics(test.y, p_test)
        rows.append(dict(reference=reference, split="validation", **valid_metrics))
        rows.append(dict(reference=reference, split="test", **test_metrics))
        test_copy = test[["user_id", "date", "y", "static_relative", "symmetric_relative",
                          "asymmetric_relative"]].reset_index(drop=True)
        test_copy["row_id"] = np.arange(len(test_copy))
        test_copy["reference"] = reference
        test_copy["predicted_reduce"] = p_test
        predictions.append(test_copy)
        for name, value in zip(names, model.coef_[0] if hasattr(model, "coef_") else np.zeros(len(names))):
            rows.append(dict(reference=reference, split="coefficient", feature=name, coefficient=float(value)))
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(predictions, ignore_index=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    # Cluster bootstrap on users, not individual rows, preserves repeated-decision dependence.
    # The same resampled user blocks are used for every reference representation,
    # so the model differences in Figure E6 are paired rather than independent.
    rng = np.random.default_rng(seed)
    boot_rows = []
    behavior_boot_rows = []
    references = ("none", "static", "symmetric", "asymmetric")
    behavior_references = references[1:]
    predictions["_user"] = predictions.user_id.astype(str)
    test["_user"] = test.user_id.astype(str)
    users = np.array(sorted(set(predictions._user) & set(test._user)), dtype=object)
    if not len(users):
        raise ValueError("No investors overlap between E6 predictions and behavior rows")
    prediction_blocks = {
        reference: {user: block[["y", "predicted_reduce"]].to_numpy(float)
                    for user, block in predictions[predictions.reference == reference].groupby("_user")}
        for reference in references
    }
    test_blocks = {user: block for user, block in test.groupby("_user")}
    for b in range(int(bootstrap)):
        sampled = rng.choice(users, size=len(users), replace=True)
        for reference in references:
            blocks = [prediction_blocks[reference].get(user) for user in sampled]
            blocks = [block for block in blocks if len(block)]
            if not blocks:
                continue
            sample = np.concatenate(blocks, axis=0)
            met = _classification_metrics(sample[:, 0].astype(int), sample[:, 1])
            boot_rows.append(dict(reference=reference, bootstrap=b, **met))
        sampled_blocks = [test_blocks.get(user) for user in sampled]
        sampled_blocks = [block for block in sampled_blocks if block is not None and len(block)]
        if sampled_blocks:
            behavior_sample = pd.concat(sampled_blocks, ignore_index=True)
            for reference in behavior_references:
                behavior_boot_rows.append(dict(
                    reference=reference, bootstrap=b,
                    **_reference_disposition_metrics(behavior_sample, reference)))
        if (b + 1) % 10 == 0:
            print(f"E6 paired bootstrap: {b + 1}/{bootstrap}", flush=True)
    boot = pd.DataFrame(boot_rows)
    intervals = []
    for reference, group in boot.groupby("reference"):
        for metric in ("log_loss", "brier", "auroc", "auprc", "calibration_slope"):
            values = group[metric].dropna()
            if len(values):
                intervals.append(dict(reference=reference, metric=metric, estimate=float(values.median()),
                                      lower=float(values.quantile(.025)), upper=float(values.quantile(.975))))
    boot.to_csv(output / "bootstrap_raw.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "bootstrap_intervals.csv", index=False)
    behavior_boot = pd.DataFrame(behavior_boot_rows)
    behavior_intervals = []
    for reference, group in behavior_boot.groupby("reference"):
        for metric in ("gain_reduction_rate", "loss_reduction_rate", "disposition_gap"):
            values = group[metric].dropna()
            if len(values):
                behavior_intervals.append(dict(reference=reference, metric=metric,
                                               estimate=float(values.median()),
                                               lower=float(values.quantile(.025)),
                                               upper=float(values.quantile(.975))))
    behavior_boot.to_csv(output / "reference_behavior_bootstrap_raw.csv", index=False)
    pd.DataFrame(behavior_intervals).to_csv(
        output / "reference_behavior_bootstrap_intervals.csv", index=False)
    pd.DataFrame([
        dict(reference=reference, **_reference_disposition_metrics(test, reference))
        for reference in behavior_references
    ]).to_csv(output / "reference_behavior_summary.csv", index=False)
    # Define test subgroups from training-period investor history only.  This
    # prevents subgroup labels from using the held-out outcomes themselves.
    user_stats = (train.groupby("_user", sort=False).y
                  .agg(historical_reduction="mean", historical_activity="size"))
    global_reduction = float(user_stats.historical_reduction.median()) if len(user_stats) else 0.0
    global_activity = float(user_stats.historical_activity.median()) if len(user_stats) else 0.0
    test_users = pd.DataFrame({"user_id": test._user.to_numpy()})
    test_users["historical_reduction"] = test_users.user_id.map(
        user_stats.historical_reduction).fillna(global_reduction)
    test_users["historical_activity"] = test_users.user_id.map(user_stats.historical_activity).fillna(global_activity)
    reduction_values = user_stats.historical_reduction.dropna().to_numpy(float) if len(user_stats) else np.array([0.0])
    activity_values = user_stats.historical_activity.to_numpy(float) if len(user_stats) else np.array([0.0])
    reduction_bins = np.quantile(reduction_values, [.333333, .666667])
    activity_bins = np.quantile(activity_values, [.333333, .666667])
    test_users["reduction_group"] = np.digitize(test_users.historical_reduction, reduction_bins)
    test_users["activity_group"] = np.digitize(test_users.historical_activity, activity_bins)
    group_labels = {0: "low", 1: "medium", 2: "high"}
    test_users["reduction_group"] = test_users.reduction_group.map(group_labels)
    test_users["activity_group"] = test_users.activity_group.map(group_labels)
    subgroup_rows = []
    for group_type in ("reduction_group", "activity_group"):
        for group_name in ("low", "medium", "high"):
            row_ids = set(test_users.index[test_users[group_type] == group_name])
            if not row_ids:
                continue
            group_scores = {}
            for reference in references:
                pred = predictions[predictions.reference == reference]
                selected = pred[pred.row_id.isin(row_ids)]
                if selected.empty:
                    continue
                met = _classification_metrics(selected.y.to_numpy(int), selected.predicted_reduce.to_numpy(float))
                group_scores[reference] = met
                subgroup_rows.append(dict(group_type=group_type, group=group_name,
                                          reference=reference, n=len(selected),
                                          log_loss=met["log_loss"], brier=met["brier"]))
            baseline = group_scores.get("none")
            if baseline is not None:
                for row in subgroup_rows:
                    if row["group_type"] == group_type and row["group"] == group_name:
                        row["log_loss_improvement"] = baseline["log_loss"] - row["log_loss"]
                        row["brier_improvement"] = baseline["brier"] - row["brier"]
    pd.DataFrame(subgroup_rows).to_csv(output / "subgroup_metrics.csv", index=False)
    manifest = dict(experiment=6, status="complete", rows=len(frame),
                    opportunities_source=str(opportunities_source) if opportunities_source else None,
                    train_end=str(train_cut.date()),
                    validation_end=str(valid_cut.date()), test_rows=len(test), references=["none", "static", "symmetric", "asymmetric"],
                    eta_gain=float(eta_gain), eta_loss=float(eta_loss),
                    primary_outcome="held-stock target weight decrease at the next rebalance",
                    behavior_statistics=["reference-conditioned gain reduction rate",
                                         "reference-conditioned loss reduction rate",
                                         "gain-minus-loss reduction gap"],
                    comparison="same decisions, controls, chronological split and logistic model; only the reference representation changes",
                    common_controls=["holding age", "event count", "previous weight", "price",
                                     "proactive", "prior investor reduction rate",
                                     "prior opportunity count", "lagged market/holding/social aggregates"],
                    bootstrap="paired investor-cluster bootstrap",
                    interpretation="supports behavioral alignment only if asymmetric improves out-of-sample decision prediction and yields a stable positive reference-conditioned disposition gap; no causal claim")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics, predictions


def plot_e6(metrics, predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    refs = ["static", "symmetric", "asymmetric"]
    labels = {"static": "Static cost", "symmetric": "Symmetric adaptive",
              "asymmetric": "Asymmetric adaptive"}
    colors = {"static": "#009E73", "symmetric": "#D55E00", "asymmetric": "#0072B2"}
    x = np.arange(len(refs))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.0), constrained_layout=True)

    boot_path = output.parent / "bootstrap_raw.csv"
    boot = pd.read_csv(boot_path) if boot_path.exists() else pd.DataFrame()
    if not boot.empty:
        pivot = boot.pivot(index="bootstrap", columns="reference", values=["log_loss", "brier"])
        width = .32
        for offset, metric, hatch, metric_label in (
                (-width / 2, "log_loss", "", "Log loss"),
                (width / 2, "brier", "//", "Brier score")):
            differences = [pivot[(metric, "none")] - pivot[(metric, ref)] for ref in refs]
            medians = np.asarray([float(values.median()) for values in differences])
            lowers = np.asarray([float(values.quantile(.025)) for values in differences])
            uppers = np.asarray([float(values.quantile(.975)) for values in differences])
            axes[0, 0].bar(x + offset, medians, width,
                           yerr=np.vstack([medians - lowers, uppers - medians]), capsize=3,
                           color=[colors[ref] for ref in refs], hatch=hatch,
                           alpha=.86, edgecolor="white", linewidth=.6)
        axes[0, 0].axhline(0, color="#555555", linewidth=.8)
        axes[0, 0].set_xticks(x, [labels[ref] for ref in refs], rotation=25, ha="right", fontsize=8)
        axes[0, 0].set_ylabel("OOS error improvement vs no reference")
        axes[0, 0].set_title("Incremental decision-prediction value")
        axes[0, 0].legend(handles=[Patch(facecolor="#999999", label="Log loss"),
                                   Patch(facecolor="#999999", hatch="//", label="Brier score")],
                          frameon=False, fontsize=8)
    else:
        axes[0, 0].text(.5, .5, "Bootstrap output unavailable", ha="center", va="center")

    behavior_path = output.parent / "reference_behavior_bootstrap_raw.csv"
    behavior = pd.read_csv(behavior_path) if behavior_path.exists() else pd.DataFrame()
    if not behavior.empty:
        width = .30
        for offset, metric, hatch in (
                (-width / 2, "gain_reduction_rate", ""),
                (width / 2, "loss_reduction_rate", "//")):
            groups = [behavior.loc[behavior.reference == ref, metric].dropna() for ref in refs]
            medians = np.asarray([float(values.median()) for values in groups])
            lowers = np.asarray([float(values.quantile(.025)) for values in groups])
            uppers = np.asarray([float(values.quantile(.975)) for values in groups])
            axes[0, 1].bar(x + offset, medians, width,
                           yerr=np.vstack([medians - lowers, uppers - medians]), capsize=3,
                           color=[colors[ref] for ref in refs], hatch=hatch,
                           alpha=.86, edgecolor="white", linewidth=.6)
        axes[0, 1].set_xticks(x, [labels[ref] for ref in refs], rotation=25, ha="right", fontsize=8)
        axes[0, 1].set_ylabel("Observed reduction rate")
        axes[0, 1].set_title("Realized behavior under each reference partition")
        axes[0, 1].legend(handles=[Patch(facecolor="#999999", label="Gain state"),
                                   Patch(facecolor="#999999", hatch="//", label="Loss state")],
                          frameon=False, fontsize=8)

        groups = [behavior.loc[behavior.reference == ref, "disposition_gap"].dropna() for ref in refs]
        medians = np.asarray([float(values.median()) for values in groups])
        lowers = np.asarray([float(values.quantile(.025)) for values in groups])
        uppers = np.asarray([float(values.quantile(.975)) for values in groups])
        axes[1, 0].bar(x, medians, yerr=np.vstack([medians - lowers, uppers - medians]),
                       capsize=4, color=[colors[ref] for ref in refs], alpha=.88,
                       edgecolor="white", linewidth=.6)
        axes[1, 0].axhline(0, color="#555555", linewidth=.8)
        axes[1, 0].set_xticks(x, [labels[ref] for ref in refs], rotation=25, ha="right", fontsize=8)
        axes[1, 0].set_ylabel("Gain minus loss reduction rate")
        axes[1, 0].set_title("Reference-conditioned disposition gap")
    else:
        axes[0, 1].text(.5, .5, "Behavior bootstrap unavailable", ha="center", va="center")
        axes[1, 0].text(.5, .5, "Behavior bootstrap unavailable", ha="center", va="center")

    subgroup_path = output.parent / "subgroup_metrics.csv"
    subgroup = pd.read_csv(subgroup_path) if subgroup_path.exists() else pd.DataFrame()
    if not subgroup.empty:
        subgroup = subgroup[subgroup.reference != "none"]
        candidates = [("reduction_group", "low"), ("reduction_group", "medium"),
                      ("reduction_group", "high"), ("activity_group", "low"),
                      ("activity_group", "medium"), ("activity_group", "high")]
        present = [(kind, value) for kind, value in candidates
                   if ((subgroup.group_type == kind) & (subgroup.group == value)).any()]
        x_subgroup = np.arange(len(present))
        width = .24
        for offset, ref in zip((-width, 0, width), refs):
            values = []
            for kind, value in present:
                row = subgroup[(subgroup.group_type == kind) & (subgroup.group == value) &
                               (subgroup.reference == ref)]
                values.append(float(row.log_loss_improvement.iloc[0]) if not row.empty else np.nan)
            axes[1, 1].bar(x_subgroup + offset, values, width,
                           label=labels[ref], color=colors[ref], alpha=.86)
        axes[1, 1].axhline(0, color="#555555", linewidth=.8)
        axes[1, 1].set_xticks(x_subgroup,
                              [f"{kind.replace('_group', '')}: {value}" for kind, value in present],
                              rotation=35, ha="right", fontsize=7)
        axes[1, 1].set_ylabel("Log-loss improvement vs no reference")
        axes[1, 1].set_title("Incremental value by pre-test investor history")
        axes[1, 1].legend(frameon=False, fontsize=7)
    else:
        axes[1, 1].text(.5, .5, "Subgroup output unavailable", ha="center", va="center")
    for ax, panel_label in zip(axes.flat, "ABCD"):
        ax.grid(alpha=.18, linewidth=.5)
        ax.set_axisbelow(True)
        _panel_label(ax, panel_label)
    fig.savefig(output / "figure_E6_behavior_alignment.pdf", bbox_inches="tight")
    fig.savefig(output / "figure_E6_behavior_alignment.svg", bbox_inches="tight")
    fig.savefig(output / "figure_E6_behavior_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _resolve_run_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def replay_e5_recommendations(e5_run, output=None):
    """Reconstruct every executed E5 portfolio from saved theta paths.

    E5 stores aggregate daily diagnostics and episode-level parameters rather
    than the 301-dimensional actions.  Execution uses a deterministic random
    stream keyed by seed and day, so the exact action path is recoverable.
    """
    e5_run = Path(e5_run)
    manifest_path = e5_run / "manifest.json"
    daily_path = e5_run / "daily.csv"
    episodes_path = e5_run / "episodes.csv"
    for path in (manifest_path, daily_path, episodes_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = _safe_json(manifest_path)
    if manifest.get("experiment") != 5 or manifest.get("status") != "complete":
        raise ValueError("E7 requires a completed E5 run")
    panel_path = _resolve_run_path(manifest["panel"])
    cache_root = _resolve_run_path(manifest["cache_root"])
    panel = PaperPanel.load(panel_path, manifest["start_date"], manifest["end_date"])
    augment_quality_feature(panel, cache_root)
    market = PaperMarket(panel, semi=False)
    config = PaperConfig(**manifest["config"])
    daily = pd.read_csv(daily_path, dtype={"date": str})
    episodes = pd.read_csv(episodes_path, dtype={"date": str})
    unknown = sorted(set(daily.method) - set(E5_METHODS))
    if unknown:
        raise ValueError(f"Unsupported E5 methods in daily output: {unknown}")
    theta_columns = [f"theta_{index}" for index in range(config.dimension)]
    missing_theta = [column for column in theta_columns if column not in episodes]
    if missing_theta:
        raise ValueError(f"E5 episode output lacks theta columns: {missing_theta}")

    records = []
    errors = {"cash": 0.0, "turnover": 0.0, "net_return": 0.0, "wealth": 0.0}
    returns_by_seed = {}
    for (seed, method), observed in daily.groupby(["seed", "method"], sort=False):
        observed = observed.sort_values("day", kind="mergesort")
        previous = np.r_[1.0, np.zeros(len(panel.codes) - 1)]
        wealth = 1.0
        returns = returns_by_seed.setdefault(int(seed), market.execution_returns(int(seed)))
        episode_rows = episodes[(episodes.seed == seed) & (episodes.method == method)]
        if method != "equal_weight":
            if episode_rows.episode.duplicated().any():
                raise ValueError(f"Duplicate E5 episode rows for method={method}, seed={seed}")
            episode_rows = episode_rows.set_index("episode")
        for row in observed.itertuples(index=False):
            day = int(row.day)
            if str(panel.dates[day]) != str(row.date):
                raise ValueError(f"E5 day/date mismatch at method={method}, seed={seed}, day={day}")
            if method == "equal_weight":
                risky_tradable = np.asarray(panel.tradable[day, 1:], dtype=bool)
                latent = np.zeros(len(previous))
                if risky_tradable.any():
                    latent[1:][risky_tradable] = 1.0 / risky_tradable.sum()
                else:
                    latent[0] = 1.0
            else:
                if int(row.episode) not in episode_rows.index:
                    raise ValueError(f"Missing E5 theta for method={method}, seed={seed}, episode={row.episode}")
                theta = episode_rows.loc[int(row.episode), theta_columns].to_numpy(float)
                features = policy_features(panel.features[day], previous, config.dimension)
                latent, _ = policy(theta, features, rng_for(int(seed), day, 40),
                                   config.policy_sharpness, config.dirichlet_scale)
            target = action_map(previous, latent, panel.tradable[day], config.trade_fraction)
            old_wealth = wealth
            wealth, _, turnover = next_wealth(wealth, previous, target, returns[day], config)
            net_return = float(wealth / old_wealth - 1.0)
            errors["cash"] = max(errors["cash"], abs(float(target[0]) - float(row.cash)))
            errors["turnover"] = max(errors["turnover"], abs(float(turnover) - float(row.turnover)))
            errors["net_return"] = max(errors["net_return"], abs(net_return - float(row.net_return)))
            errors["wealth"] = max(errors["wealth"], abs(float(wealth) - float(row.wealth)))
            risky = target[1:]
            risky_total = float(risky.sum())
            effective = (risky_total ** 2 / float(risky @ risky)
                         if float(risky @ risky) > 0 else 0.0)
            records.append(dict(seed=int(seed), method=method, episode=int(row.episode),
                                day=day, date=str(row.date), weights=target.tolist(),
                                cash=float(target[0]), risky_weight=risky_total,
                                effective_holdings=effective,
                                max_stock_weight=float(risky.max(initial=0.0)),
                                advisor_turnover=float(turnover), net_return=net_return))
            previous = target
    tolerance = 1e-10
    if max(errors.values(), default=0.0) > tolerance:
        raise ValueError(f"E5 recommendation replay failed validation: {errors}")
    recommendations = pd.DataFrame(records)
    metadata = dict(source_run=str(e5_run.resolve()), panel=str(panel_path.resolve()),
                    panel_sha256=manifest.get("panel_sha256"), codes=panel.codes,
                    methods=sorted(recommendations.method.unique()),
                    seeds=sorted(int(seed) for seed in recommendations.seed.unique()),
                    advisor_eta_gain=float(config.eta_gain),
                    advisor_eta_loss=float(config.eta_loss),
                    validation_max_absolute_error=errors,
                    rows=len(recommendations), assets=len(panel.codes) - 1)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_parquet(output, index=False)
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return recommendations, panel, config, metadata


def _describe_agent_clusters(cluster_profile):
    names = {}
    disposition_column = ("e6_disposition" if "e6_disposition" in cluster_profile
                          else "disposition")
    for cluster, row in cluster_profile.iterrows():
        disposition = cluster_profile[disposition_column]
        activity = cluster_profile.transnum
        diversification = cluster_profile.stknum
        if row[disposition_column] >= disposition.quantile(.67):
            names[cluster] = "Disposition-effect investors"
        elif row.transnum <= activity.quantile(.33):
            names[cluster] = "Low-frequency diversified investors"
        elif (row.transnum >= activity.quantile(.67) and
              row.stknum >= diversification.quantile(.67)):
            names[cluster] = "Active diversified investors"
        elif row.stknum <= diversification.quantile(.33):
            names[cluster] = "Moderate-frequency focused investors"
        else:
            names[cluster] = "Balanced investors"
    # Ties can make two descriptions identical; retain the behavioral meaning
    # while adding a readable cohort suffix rather than an opaque type_x label.
    used = {}
    for cluster in list(names):
        base = names[cluster]
        used[base] = used.get(base, 0) + 1
        if used[base] > 1:
            names[cluster] = f"{base} (cohort {used[base]})"
    return names


def _summarize_e6_users(frame, priors, strength=20.0):
    base = frame.groupby("_user", sort=False).y.agg(reduced="sum", opportunities="size")
    gain = (frame[frame.asymmetric_gain >= .5].groupby("_user", sort=False).y
            .agg(gain_reduced="sum", gain_opportunities="size"))
    loss = (frame[frame.asymmetric_gain < .5].groupby("_user", sort=False).y
            .agg(loss_reduced="sum", loss_opportunities="size"))
    result = base.join(gain, how="left").join(loss, how="left").fillna(0.0)
    result["e6_pgr"] = ((result.gain_reduced + strength * priors["gain"])
                         / (result.gain_opportunities + strength))
    result["e6_plr"] = ((result.loss_reduced + strength * priors["loss"])
                         / (result.loss_opportunities + strength))
    result["e6_reduce_rate"] = ((result.reduced + strength * priors["overall"])
                                 / (result.opportunities + strength))
    result["e6_disposition"] = result.e6_pgr - result.e6_plr
    result["e6_opportunity_count"] = result.opportunities
    return result[["e6_pgr", "e6_plr", "e6_disposition", "e6_reduce_rate",
                   "e6_opportunity_count", "gain_opportunities", "loss_opportunities"]]


def _load_e6_user_calibration(path, train_end, shrinkage=20.0):
    """Build leakage-free, shrunk investor parameters from E6 opportunities."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    frame["_user"] = frame.user_id.map(_normalise_code)
    frame["y"] = pd.to_numeric(frame.y, errors="coerce")
    frame["asymmetric_gain"] = pd.to_numeric(frame.asymmetric_gain, errors="coerce")
    frame = frame.dropna(subset=["date", "y", "asymmetric_gain"])
    frame = frame[frame._user != ""]
    train_cut = pd.Timestamp(train_end)
    train = frame[frame.date < train_cut].copy()
    holdout = frame[frame.date >= train_cut].copy()
    if train.empty or holdout.empty:
        raise ValueError("E6 investor calibration requires nonempty training and held-out periods")
    gain = train.asymmetric_gain >= .5
    priors = {"gain": float(train.loc[gain, "y"].mean()),
              "loss": float(train.loc[~gain, "y"].mean()),
              "overall": float(train.y.mean())}
    train_stats = _summarize_e6_users(train, priors, shrinkage)
    holdout_stats = _summarize_e6_users(holdout, priors, shrinkage)
    metadata = dict(train_end=str(train_cut.date()), shrinkage=float(shrinkage),
                    training_rows=len(train), holdout_rows=len(holdout), priors=priors)
    return train_stats, holdout_stats, metadata


def _percentile_rank(values):
    values = pd.to_numeric(values, errors="coerce")
    return values.rank(method="average", pct=True).fillna(.5)


def fit_agent_types(behavior_root, output, clusters=4, seed=2026,
                    e6_opportunities=None, e6_model=None, shrinkage=20.0):
    behavior_root, output = Path(behavior_root), Path(output)
    if e6_opportunities is None or e6_model is None:
        raise ValueError("E7 requires both E6 opportunities and the completed E6 model directory")
    e6_model = Path(e6_model)
    e6_manifest = _safe_json(e6_model / "manifest.json")
    if e6_manifest.get("experiment") != 6 or e6_manifest.get("status") != "complete":
        raise ValueError("E7 requires a completed E6 model manifest")
    metrics_path = e6_model / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    opportunities_meta = _safe_json(Path(e6_opportunities).with_suffix(".json"))
    eta_gain = float(e6_manifest["eta_gain"])
    eta_loss = float(e6_manifest["eta_loss"])
    if opportunities_meta:
        if (abs(float(opportunities_meta.get("eta_gain", eta_gain)) - eta_gain) > 1e-12 or
                abs(float(opportunities_meta.get("eta_loss", eta_loss)) - eta_loss) > 1e-12):
            raise ValueError("E6 opportunity and model eta values do not match")
    e6_train, e6_holdout, e6_metadata = _load_e6_user_calibration(
        e6_opportunities, e6_manifest["train_end"], shrinkage)

    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_pickle(behavior_root / "regression_afterholiday.pkl")
    source["date"] = pd.to_datetime(source["day"], errors="coerce")
    source = source.sort_values(["zhuliren", "date"], kind="mergesort").copy()
    for column in ("pos", "neg", "neu", "num_posts"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
        source[f"lag_{column}"] = source.groupby("zhuliren")[column].shift(1)
    dates = np.array(sorted(source.date.dropna().dt.normalize().unique()))
    if len(dates) < 3:
        raise ValueError("Behavior table has too few dated observations")
    train_cut = dates[int(.70 * len(dates))]
    train = source[source.date <= train_cut].copy()
    stats_columns = ["PGR", "PLR", "disposition", "transnum", "stknum", "num_posts",
                     "lag_pos", "lag_neg", "lag_neu", "lag_num_posts"]
    for col in stats_columns:
        train[col] = pd.to_numeric(train[col], errors="coerce")
    stats = train.groupby("zhuliren")[stats_columns].mean()
    stats.index = stats.index.map(_normalise_code)
    stats = stats[stats.index != ""].groupby(level=0).mean()
    profile_path = behavior_root / "zlr_profile.xlsx"
    if profile_path.exists():
        profile = pd.read_excel(profile_path)
        profile["user_id"] = profile.user_id.map(_normalise_code)
        profile_cols = [c for c in ("guanzhu", "fans", "status_count") if c in profile]
        if profile_cols:
            profile = profile[["user_id", *profile_cols]].drop_duplicates("user_id").set_index("user_id")
            stats = stats.join(profile, how="left")
    stocks_path = behavior_root / "zhulirenstocks_count.xlsx"
    if stocks_path.exists():
        stock_count = pd.read_excel(stocks_path)
        stock_count["user_id"] = stock_count.user_id.map(_normalise_code)
        stock_count = stock_count[["user_id", "stocks_count"]].drop_duplicates("user_id").set_index("user_id")
        stats = stats.join(stock_count, how="left")
    follow_path = behavior_root / "tm_zhulirenguanzhu.pkl"
    if follow_path.exists():
        follows = pd.read_pickle(follow_path)
        stats["following_count"] = [len(follows.get(int(user), follows.get(user, [])))
                                    if str(user).isdigit() else 0 for user in stats.index]
    else:
        stats["following_count"] = 0

    stats = stats.join(e6_train, how="inner")
    if len(stats) < clusters:
        raise ValueError(f"Only {len(stats)} calibrated users available for {clusters} clusters")
    stats["sentiment_balance"] = ((stats.lag_pos - stats.lag_neg)
                                  / (stats.lag_pos.abs() + stats.lag_neg.abs() + stats.lag_neu.abs() + 1.0))
    stats["activity_rank"] = _percentile_rank(stats.transnum)
    stats["diversification_rank"] = _percentile_rank(stats.stknum)
    stats["reference_rank"] = _percentile_rank(stats.e6_disposition)
    stats["base_adoption"] = stats.e6_reduce_rate.clip(.05, .95)
    stats["loss_aversion"] = (1.0 + 4.0 * stats.e6_disposition.clip(lower=0.0)).clip(1.0, 4.0)
    cluster_columns = ["e6_pgr", "e6_plr", "e6_disposition", "e6_reduce_rate",
                       "transnum", "stknum", "sentiment_balance", "lag_num_posts",
                       "following_count"]
    numeric = stats[cluster_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing = numeric.isna().astype(float)
    missing.columns = [f"{column}_missing" for column in missing]
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    for column in ("transnum", "stknum", "lag_num_posts", "following_count"):
        numeric[column] = np.log1p(np.maximum(numeric[column], 0.0))
    design = pd.concat([numeric, missing.loc[:, missing.any(axis=0)]], axis=1)
    scaler = StandardScaler().fit(design)
    labels = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit_predict(scaler.transform(design))
    stats["cluster"] = labels
    cluster_profile = stats.groupby("cluster").mean(numeric_only=True)
    names = _describe_agent_clusters(cluster_profile)
    stats["agent_type"] = stats.cluster.map(names)
    counts = stats.agent_type.value_counts()
    stats["type_prevalence"] = stats.agent_type.map(counts / len(stats))
    assignments = stats.reset_index(names="user_id")
    assignments.to_csv(output / "agent_assignments.csv", index=False)
    profile = stats.groupby("agent_type").mean(numeric_only=True)
    profile["type_count"] = counts.reindex(profile.index).astype(int)
    profile["type_prevalence"] = profile.type_count / profile.type_count.sum()
    profile.reset_index().to_csv(output / "agent_type_profiles.csv", index=False)

    validation_rows = []
    behavior_holdout = source[source.date > train_cut].copy()
    for col in stats_columns:
        behavior_holdout[col] = pd.to_numeric(behavior_holdout[col], errors="coerce")
    behavior_holdout = behavior_holdout.groupby("zhuliren")[stats_columns].mean()
    behavior_holdout.index = behavior_holdout.index.map(_normalise_code)
    behavior_holdout = behavior_holdout.groupby(level=0).mean()
    for validation_source, frame in (("behavior", behavior_holdout), ("e6", e6_holdout)):
        frame = frame.copy()
        frame["agent_type"] = frame.index.map(stats.agent_type.to_dict())
        frame = frame.dropna(subset=["agent_type"])
        for agent_type, group in frame.groupby("agent_type"):
            for column in group.columns.difference(["agent_type"]):
                if column not in profile or not pd.api.types.is_numeric_dtype(group[column]):
                    continue
                observed = float(pd.to_numeric(group[column], errors="coerce").mean())
                train_value = float(profile.loc[agent_type, column])
                validation_rows.append(dict(source=validation_source, agent_type=agent_type,
                                            feature=column, train_profile=train_value,
                                            holdout_observed=observed,
                                            absolute_gap=abs(observed - train_value)))
    pd.DataFrame(validation_rows).to_csv(output / "agent_type_holdout_validation.csv", index=False)
    calibration_manifest = dict(source="behavior/regression_afterholiday.pkl + profile/network tables",
                                e6_opportunities=str(Path(e6_opportunities).resolve()),
                                e6_model=str(e6_model.resolve()),
                                e6_training_calibration=True,
                                e6_eta_gain=eta_gain, e6_eta_loss=eta_loss,
                                e6_user_calibration=e6_metadata,
                                e6_global_coefficients_used_for_clustering=False,
                                cluster_features=cluster_columns,
                                clusters=clusters, calibrated_users=len(stats),
                                agent_type_names=names,
                                behavior_train_cut=str(pd.Timestamp(train_cut).date()),
                                social_timestamp_available=False,
                                social_timestamp_note="lagged daily aggregates and static network counts only")
    (output / "calibration_manifest.json").write_text(
        json.dumps(calibration_manifest, indent=2), encoding="utf-8")
    return assignments, profile


def _prospect_value(relative, loss_aversion, alpha=.88, epsilon=.01):
    magnitude = (relative * relative + epsilon * epsilon) ** (alpha / 2.0) - epsilon ** alpha
    return magnitude if relative >= 0 else -loss_aversion * magnitude


def _array_ranks(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def _effective_holdings(weights):
    risky = np.asarray(weights, float)[1:]
    total = float(risky.sum())
    square = float(risky @ risky)
    return total * total / square if square > 0 else 0.0


def _sample_calibrated_agents(assignments, agents_per_type, rng):
    required = ["user_id", "agent_type", "activity_rank", "diversification_rank",
                "base_adoption", "loss_aversion", "e6_pgr", "e6_plr", "e6_disposition"]
    missing = [column for column in required if column not in assignments]
    if missing:
        raise ValueError(f"Agent assignments lack calibrated fields: {missing}")
    samples = []
    for agent_type, group in assignments.groupby("agent_type", sort=True):
        count = min(int(agents_per_type), len(group))
        if count < 1:
            continue
        chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
        sample = group.loc[chosen].copy()
        sample["simulation_agent"] = [f"agent_{len(samples):02d}_{index:03d}" for index in range(count)]
        samples.append(sample)
    if not samples:
        raise ValueError("No calibrated investors available for E7 simulation")
    return pd.concat(samples, ignore_index=True)


def simulate_agents(recommendations, assignments, profiles, panel, config, output,
                    investor_eta_gain, investor_eta_loss, seed=2026,
                    agents_per_type=16, source_metadata=None):
    """Run leakage-free, horizon-aligned recommendation interactions.

    Adoption is decided once at the start of each E5 episode, before any return
    in that episode is observed.  Satisfaction is measured after the full
    horizon against a no-rebalance counterfactual that starts from the same
    wealth, reference and holdings.
    """
    output = Path(output)
    recommendations = recommendations.copy()
    recommendations = recommendations[recommendations.method.isin(E5_METHODS)].copy()
    recommendations["date"] = recommendations.date.astype(str)
    if set(recommendations.method.unique()) != set(E5_METHODS):
        raise ValueError("E7 requires the same six recommendation methods as formal E5")
    if not (0 <= investor_eta_gain < 1 and 0 <= investor_eta_loss < 1):
        raise ValueError("Investor reference adaptation rates must lie in [0,1)")
    rng = np.random.default_rng(seed)
    agents = _sample_calibrated_agents(assignments, agents_per_type, rng)
    profile = profiles.copy()
    if "agent_type" in profile.columns:
        profile = profile.set_index("agent_type")
    type_weights = profile.type_prevalence.to_dict()
    market = PaperMarket(panel, semi=False)
    rows = []
    method_order = [method for method in E5_METHODS if method in set(recommendations.method)]

    for agent in agents.itertuples(index=False):
        preferred_gain_share = float(agent.e6_pgr / max(agent.e6_pgr + agent.e6_plr, 1e-12))
        for market_seed, seed_rows in recommendations.groupby("seed", sort=True):
            execution_returns = market.execution_returns(int(market_seed))
            states = {
                method: dict(wealth=1.0, reference=1.0,
                             previous=np.r_[1.0, np.zeros(len(panel.codes) - 1)],
                             basis=np.ones(len(panel.codes)), lag_satisfaction=0.0,
                             cumulative_cost=0.0)
                for method in method_order
            }
            prices = np.ones(len(panel.codes))
            for episode, episode_rows in seed_rows.groupby("episode", sort=True):
                episode_rows = episode_rows.sort_values(["day", "method"], kind="mergesort")
                episode_days = sorted(episode_rows.day.astype(int).unique())
                if len(episode_days) != config.horizon:
                    raise ValueError(
                        f"E7 episode length mismatch for seed={market_seed}, episode={episode}: "
                        f"expected {config.horizon}, got {len(episode_days)}"
                    )
                first_day = episode_days[0]
                first_rows = episode_rows[episode_rows.day.eq(first_day)].set_index("method")
                if any(method not in first_rows.index for method in method_order):
                    raise ValueError(
                        f"Incomplete E5 recommendations for seed={market_seed}, episode={episode}"
                    )
                first_tradable = panel.tradable[first_day]
                turnover_values, diversification_values, gain_sale_values = [], [], []
                for method in method_order:
                    state = states[method]
                    target = np.asarray(first_rows.loc[method, "weights"], float)
                    target = executable_target(state["previous"], target, first_tradable)
                    turnover = .5 * float(np.abs(target - state["previous"]).sum())
                    sales = np.maximum(state["previous"] - target, 0.0)
                    held = state["previous"] > 1e-12
                    gains = (prices >= state["basis"]) & held
                    risky_sales = sales[1:]
                    total_sales = float(risky_sales.sum())
                    gain_sale_share = (float(risky_sales[gains[1:]].sum()) / total_sales
                                       if total_sales > 1e-12 else .5)
                    turnover_values.append(turnover)
                    diversification_values.append(_effective_holdings(target))
                    gain_sale_values.append(gain_sale_share)
                turnover_ranks = _array_ranks(turnover_values)
                diversification_ranks = _array_ranks(diversification_values)
                adoption_draw = float(rng.random())
                interactions = {}
                for method_index, method in enumerate(method_order):
                    state = states[method]
                    turnover_match = 1.0 - abs(float(turnover_ranks[method_index]) - float(agent.activity_rank))
                    diversification_match = 1.0 - abs(float(diversification_ranks[method_index]) -
                                                       float(agent.diversification_rank))
                    disposition_match = 1.0 - abs(float(gain_sale_values[method_index]) - preferred_gain_share)
                    compatibility = float(np.clip(.35 * turnover_match + .25 * diversification_match +
                                                  .40 * disposition_match, 0.0, 1.0))
                    base = float(np.clip(agent.base_adoption, 1e-5, 1 - 1e-5))
                    logit_base = math.log(base / (1.0 - base))
                    adoption_probability = float(expit(logit_base + 2.0 * (compatibility - .5) +
                                                       .35 * np.tanh(state["lag_satisfaction"] / 25.0)))
                    adopted = int(adoption_draw < adoption_probability)
                    interactions[method] = dict(
                        adopted=adopted, adoption_probability=adoption_probability,
                        compatibility=compatibility, turnover_match=turnover_match,
                        diversification_match=diversification_match,
                        disposition_match=disposition_match,
                        starting_wealth=float(state["wealth"]),
                        status_wealth=float(state["wealth"]),
                        status_reference=float(state["reference"]),
                        status_previous=state["previous"].copy(),
                        total_turnover=0.0, total_cost=0.0,
                    )

                for day in episode_days:
                    day_rows = episode_rows[episode_rows.day.eq(day)].set_index("method")
                    if any(method not in day_rows.index for method in method_order):
                        raise ValueError(f"Incomplete E5 recommendations for seed={market_seed}, day={day}")
                    returns = execution_returns[day]
                    tradable = panel.tradable[day]
                    for method in method_order:
                        state = states[method]
                        interaction = interactions[method]
                        previous = state["previous"]
                        if interaction["adopted"]:
                            proposed = np.asarray(day_rows.loc[method, "weights"], float)
                            executed = executable_target(previous, proposed, tradable)
                        else:
                            executed = previous.copy()
                        turnover_l1 = float(np.abs(executed - previous).sum())
                        turnover = .5 * turnover_l1
                        fee_fraction = config.cost * turnover_l1
                        executed_multiplier = 1.0 + float(executed @ returns) - fee_fraction
                        status_multiplier = 1.0 + float(interaction["status_previous"] @ returns)
                        if status_multiplier <= 0 or executed_multiplier <= 0:
                            raise FloatingPointError("Nonpositive E7 wealth multiplier")

                        wealth_before = float(state["wealth"])
                        state["wealth"] = wealth_before * executed_multiplier
                        reference_before = float(state["reference"])
                        eta = (investor_eta_gain if state["wealth"] >= reference_before
                               else investor_eta_loss)
                        state["reference"] = reference_before + eta * (state["wealth"] - reference_before)

                        interaction["status_wealth"] *= status_multiplier
                        status_reference_before = float(interaction["status_reference"])
                        status_eta = (investor_eta_gain
                                      if interaction["status_wealth"] >= status_reference_before
                                      else investor_eta_loss)
                        interaction["status_reference"] = (
                            status_reference_before + status_eta *
                            (interaction["status_wealth"] - status_reference_before)
                        )

                        if interaction["adopted"]:
                            buys = np.maximum(executed - previous, 0.0)
                            denominator = previous + buys
                            updated_basis = state["basis"].copy()
                            bought = buys > 1e-12
                            updated_basis[bought] = ((previous[bought] * state["basis"][bought] +
                                                      buys[bought] * prices[bought]) /
                                                     denominator[bought])
                            updated_basis[executed <= 1e-12] = prices[executed <= 1e-12]
                            state["basis"] = updated_basis
                        state["previous"] = executed
                        interaction["total_turnover"] += turnover
                        cost = wealth_before * fee_fraction
                        interaction["total_cost"] += cost
                        state["cumulative_cost"] += cost
                    prices = prices * (1.0 + returns)

                for method in method_order:
                    state = states[method]
                    interaction = interactions[method]
                    executed_value = _prospect_value(
                        state["wealth"] - state["reference"], float(agent.loss_aversion))
                    status_value = _prospect_value(
                        interaction["status_wealth"] - interaction["status_reference"],
                        float(agent.loss_aversion))
                    satisfaction = float(
                        (executed_value - status_value) /
                        max(interaction["starting_wealth"], 1e-12)
                    )
                    if not interaction["adopted"] and abs(satisfaction) > 1e-12:
                        raise AssertionError("Rejected E7 recommendation must equal the hold counterfactual")
                    satisfaction_x1e4 = satisfaction * 10000.0
                    state["lag_satisfaction"] = (
                        .8 * state["lag_satisfaction"] + .2 * satisfaction_x1e4
                    )
                    rows.append(dict(simulation_agent=agent.simulation_agent,
                                     source_user_id=agent.user_id, agent_type=agent.agent_type,
                                     type_prevalence=float(type_weights[agent.agent_type]),
                                     method=method, seed=int(market_seed), episode=int(episode),
                                     start_day=first_day, end_day=episode_days[-1],
                                     start_date=str(first_rows.loc[method, "date"]),
                                     end_date=str(episode_rows[
                                         (episode_rows.day.eq(episode_days[-1])) &
                                         (episode_rows.method.eq(method))].iloc[0].date),
                                     horizon_days=len(episode_days),
                                     adoption_probability=interaction["adoption_probability"],
                                     adopted=interaction["adopted"],
                                     compatibility=interaction["compatibility"],
                                     turnover_match=interaction["turnover_match"],
                                     diversification_match=interaction["diversification_match"],
                                     disposition_match=interaction["disposition_match"],
                                     satisfaction=satisfaction,
                                     satisfaction_x1e4=satisfaction_x1e4,
                                     wealth=state["wealth"], reference=state["reference"],
                                     status_quo_wealth=interaction["status_wealth"],
                                     status_quo_reference=interaction["status_reference"],
                                     episode_turnover=interaction["total_turnover"],
                                     episode_transaction_cost=interaction["total_cost"],
                                     calibrated_pgr=float(agent.e6_pgr),
                                     calibrated_plr=float(agent.e6_plr),
                                     calibrated_disposition=float(agent.e6_disposition)))
    traces = pd.DataFrame(rows)
    if traces.empty:
        raise ValueError("E7 simulation produced no interaction rows")
    path = (traces.sort_values(["simulation_agent", "method", "seed", "episode"])
            .groupby(["agent_type", "simulation_agent", "method", "seed"], as_index=False)
            .agg(terminal_wealth=("wealth", "last"),
                 cumulative_interaction_value=("satisfaction", "sum"),
                 total_turnover=("episode_turnover", "sum"),
                 cumulative_cost=("episode_transaction_cost", "sum")))
    summary = (traces.groupby(["agent_type", "method"], as_index=False)
               .agg(adoption_rate=("adopted", "mean"),
                    mean_adoption_probability=("adoption_probability", "mean"),
                    mean_interaction_value_x1e4=("satisfaction_x1e4", "mean"),
                    mean_compatibility=("compatibility", "mean")))
    conditional = (traces[traces.adopted.eq(1)]
                   .groupby(["agent_type", "method"], as_index=False)
                   .agg(mean_conditional_satisfaction_x1e4=("satisfaction_x1e4", "mean"),
                        adopted_episodes=("adopted", "size")))
    summary = summary.merge(conditional, on=["agent_type", "method"], how="left",
                            validate="one_to_one")
    path_summary = (path.groupby(["agent_type", "method"], as_index=False)
                    .agg(mean_terminal_wealth=("terminal_wealth", "mean"),
                         mean_cumulative_interaction_value=("cumulative_interaction_value", "mean"),
                         mean_total_turnover=("total_turnover", "mean"),
                         mean_cumulative_cost=("cumulative_cost", "mean")))
    summary = summary.merge(path_summary, on=["agent_type", "method"], validate="one_to_one")
    summary["type_prevalence"] = summary.agent_type.map(type_weights)
    weighted_rows = []
    metrics = ["adoption_rate", "mean_adoption_probability",
               "mean_conditional_satisfaction_x1e4", "mean_interaction_value_x1e4",
               "mean_compatibility", "mean_terminal_wealth",
               "mean_cumulative_interaction_value", "mean_total_turnover",
               "mean_cumulative_cost"]
    for method, group in summary.groupby("method", sort=False):
        weights = group.type_prevalence.to_numpy(float)
        weights = weights / weights.sum()
        weighted_rows.append(dict(method=method, **{
            metric: float(np.average(group[metric], weights=weights)) for metric in metrics
        }))
    overall = pd.DataFrame(weighted_rows)
    traces.to_parquet(output / "agent_traces.parquet", index=False)
    summary.to_csv(output / "agent_method_summary.csv", index=False)
    overall.to_csv(output / "method_overall_summary.csv", index=False)
    profile.reset_index().to_csv(output / "agent_type_profiles_simulated.csv", index=False)
    manifest = dict(experiment=7, status="complete", methods=method_order,
                    simulated_agents=len(agents), agents_per_type=int(agents_per_type),
                    advisor_eta_gain=float((source_metadata or {}).get("advisor_eta_gain", np.nan)),
                    advisor_eta_loss=float((source_metadata or {}).get("advisor_eta_loss", np.nan)),
                    investor_eta_gain=float(investor_eta_gain),
                    investor_eta_loss=float(investor_eta_loss),
                    interaction_horizon=int(config.horizon), adoption_frequency="once per E5 episode",
                    loop="episode-start recommendation -> compatibility -> simulated adoption -> h-day execution -> terminal satisfaction/reference update",
                    adoption_information="episode-start recommendation, calibrated history and lagged satisfaction only; all episode returns excluded",
                    satisfaction="conditional h-day normalized prospect-value improvement versus a no-rebalance counterfactual with its own reference path",
                    interaction_value="adoption-weighted satisfaction with rejected recommendations contributing zero",
                    paired_market_paths="seed-level E5 paths and common adoption draws retained across methods",
                    type_aggregation="overall results weighted by empirical calibrated-user prevalence",
                    recommendation="full cash-plus-300-stock E5 portfolio action",
                    limitation="simulated adoption and satisfaction are calibrated counterfactual quantities, not observed user outcomes")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary, traces, overall


def plot_e7(summary, profiles, output, overall=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    order_types = sorted(summary.agent_type.unique())
    order_methods = [m for m in E5_METHODS if m in set(summary.method)]
    method_labels = [METHOD_LABELS[m] for m in order_methods]
    method_colors = [PLOT_COLORS[m] for m in order_methods]
    with plt.rc_context({"font.family": "DejaVu Sans",
                         "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
        profile_cols = [c for c in ("e6_pgr", "e6_plr", "e6_disposition", "transnum", "stknum",
                                    "sentiment_balance", "following_count") if c in profiles]
        profile_frame = profiles.set_index("agent_type").reindex(order_types)[profile_cols].fillna(0.0)
        profile_values = StandardScaler().fit_transform(profile_frame) if profile_cols else np.zeros((len(order_types), 1))
        im = axes[0, 0].imshow(profile_values, aspect="auto", cmap="RdBu_r")
        axes[0, 0].set_xticks(np.arange(len(profile_cols)), profile_cols, rotation=35, ha="right")
        axes[0, 0].set_yticks(np.arange(len(order_types)), order_types, fontsize=7)
        axes[0, 0].set_title("Calibrated investor profiles")
        fig.colorbar(im, ax=axes[0, 0], fraction=.046, pad=.04)
        for ax, value, title, fmt in (
                (axes[0, 1], "adoption_rate", "Episode-level adoption by investor type", ".2f"),
                (axes[1, 0], "mean_conditional_satisfaction_x1e4",
                 "Five-day satisfaction after adoption", ".1f")):
            matrix = summary.pivot(index="agent_type", columns="method", values=value).reindex(index=order_types, columns=order_methods)
            im = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="YlGn")
            ax.set_xticks(np.arange(len(order_methods)), method_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(np.arange(len(order_types)), order_types, fontsize=7)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix.iloc[i, j]):
                        ax.text(j, i, format(matrix.iloc[i, j], fmt), ha="center", va="center", fontsize=7)

        if overall is None:
            weighted = summary.copy()
            weighted["weighted_adoption"] = weighted.adoption_rate * weighted.type_prevalence
            weighted["weighted_satisfaction"] = (
                weighted.mean_conditional_satisfaction_x1e4 * weighted.type_prevalence
            )
            weighted["weighted_compatibility"] = weighted.mean_compatibility * weighted.type_prevalence
            average = (weighted.groupby("method", as_index=False)
                       .agg(adoption_rate=("weighted_adoption", "sum"),
                            mean_conditional_satisfaction_x1e4=("weighted_satisfaction", "sum"),
                            mean_compatibility=("weighted_compatibility", "sum")))
        else:
            average = overall.copy()
        short_labels = {"dynamic": "DRCPT", "symmetric": "Symmetric", "static": "Static",
                        "expected": "Expected", "exponential": "Exponential",
                        "equal_weight": "Equal Weight"}
        for _, row in average.iterrows():
            method = row.method
            compatibility = np.clip(row.mean_compatibility, 0.0, 1.0)
            axes[1, 1].scatter(row.adoption_rate, row.mean_conditional_satisfaction_x1e4,
                               s=220 + 780 * compatibility, color=PLOT_COLORS[method],
                               alpha=.86, edgecolor="white", linewidth=.8)
            axes[1, 1].annotate(short_labels[method],
                                (row.adoption_rate, row.mean_conditional_satisfaction_x1e4),
                                xytext=(5, 4), textcoords="offset points", fontsize=7)
        axes[1, 1].set_xlabel("Average simulated adoption rate")
        axes[1, 1].set_ylabel(r"Five-day satisfaction after adoption ($\times 10^4$)")
        axes[1, 1].set_title("Method-level adoption--satisfaction trade-off")
        for ax, panel_label in zip(axes.flat, "ABCD"):
            ax.grid(alpha=.16, linewidth=.5)
            ax.set_axisbelow(True)
            _panel_label(ax, panel_label)
        fig.savefig(output / "figure_E7_heterogeneous_agents.pdf", bbox_inches="tight")
        fig.savefig(output / "figure_E7_heterogeneous_agents.svg", bbox_inches="tight")
        fig.savefig(output / "figure_E7_heterogeneous_agents.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
