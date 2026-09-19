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

from .paper_experiments import (PaperConfig, action_map, next_wealth, run_online)
from .paper_market import FEATURES, PaperMarket, PaperPanel, file_hash, prepare_panel


METHOD_LABELS = {
    "dynamic": "DRCPT-PG",
    "symmetric": "Symmetric CPT-PG",
    "static": "Static CPT-PG",
    "expected": "Expected-Wealth PG",
    "exponential": "Exponential-Utility PG",
    "equal_weight": "Equal Weight",
    "hs300_index": "HS300 Index Weight",
    "inverse_volatility": "Inverse Volatility",
}
E5_METHODS = ["dynamic", "symmetric", "static", "expected", "exponential",
              "equal_weight", "hs300_index", "inverse_volatility"]
PLOT_COLORS = {
    "dynamic": "#0072B2", "symmetric": "#D55E00", "static": "#009E73",
    "expected": "#CC79A7", "exponential": "#E69F00", "equal_weight": "#666666",
    "hs300_index": "#56B4E9", "inverse_volatility": "#000000",
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

    The existing panel has a neutral placeholder in that slot.  This function
    replaces it only with records whose announcement date is strictly before
    the trading date.  Missing fundamentals are neutral and reported in the
    returned coverage table rather than filled from the future.
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
    fundamentals = (fundamentals.dropna(subset=["ann_date"])
                    .sort_values(["ts_code", "ann_date"])
                    .drop_duplicates(["ts_code", "ann_date"], keep="last"))

    coverage = []
    code_index = {code: i for i, code in enumerate(panel.codes[1:], start=1)}
    for day_index, date in enumerate(panel.dates):
        date = str(date)
        known = fundamentals[fundamentals.ann_date < date]
        latest = (known.sort_values("ann_date")
                  .drop_duplicates("ts_code", keep="last")
                  .set_index("ts_code"))
        quality = latest.reindex(panel.codes[1:])[["roe", "roa", "roic"]].mean(axis=1)
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


def load_hs300_weights(cache_root, codes, dates):
    """Load the latest cached HS300 snapshot available on each date."""
    cache_root = Path(cache_root)
    frames = []
    for payload, path in _cache_records(cache_root, "index_weight"):
        pkl = path.with_suffix(".pkl")
        if payload.get("index_code") != "000300.SH" or not pkl.exists():
            continue
        frame = pd.read_pickle(pkl).copy()
        required = {"trade_date", "con_code", "weight"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[["trade_date", "con_code", "weight"]]
        frame["trade_date"] = frame.trade_date.astype(str)
        frame["con_code"] = frame.con_code.map(_normalise_code)
        frame["weight"] = pd.to_numeric(frame.weight, errors="coerce")
        frames.append(frame.dropna(subset=["weight"]))
    if not frames:
        raise FileNotFoundError(f"No usable HS300 index_weight cache under {cache_root}")
    source = pd.concat(frames, ignore_index=True)
    source = source[source.con_code.isin(codes[1:])]
    source = source.groupby(["trade_date", "con_code"], as_index=False).weight.last()
    snapshots = {}
    for trade_date, group in source.groupby("trade_date"):
        weights = group.set_index("con_code").weight.reindex(codes[1:]).fillna(0.0).to_numpy()
        total = weights.sum()
        if total > 0:
            snapshots[trade_date] = weights / total
    available = sorted(snapshots)
    result = []
    for date in map(str, dates):
        candidates = [d for d in available if d <= date]
        snapshot_date = candidates[-1] if candidates else None
        result.append((snapshot_date, snapshots.get(snapshot_date)))
    return result


def _target_from_signal(signal, tradable):
    signal = np.asarray(signal, dtype=float)
    target = np.zeros(len(signal) + 1)
    target[0] = 1.0 if not np.any(tradable) else 0.0
    if np.any(tradable):
        values = np.where(tradable, np.maximum(signal, 0.0), 0.0)
        total = values.sum()
        if total > 0:
            target[1:] = values / total
        else:
            target[0] = 1.0
    return target


def _benchmark_target(name, day, panel, previous, index_weights):
    tradable = np.asarray(panel.tradable[day, 1:], dtype=bool)
    if name == "equal_weight":
        return _target_from_signal(np.ones(len(tradable)), tradable)
    if name == "hs300_index":
        _, weights = index_weights[day]
        return _target_from_signal(np.zeros(len(tradable)) if weights is None else weights, tradable)
    if name == "inverse_volatility":
        left = max(0, day - 60)
        history = panel.returns[left:day, 1:]
        volatility = np.nanstd(history, axis=0, ddof=1) if len(history) > 1 else np.ones(len(tradable))
        signal = 1.0 / np.maximum(np.nan_to_num(volatility, nan=np.inf, posinf=np.inf), 1e-4)
        return _target_from_signal(signal, tradable)
    raise ValueError(name)


def run_benchmarks(market: PaperMarket, config: PaperConfig, seeds, methods, start_day, steps,
                   index_weights):
    rows = []
    for seed in seeds:
        returns = market.execution_returns(seed)
        for method in methods:
            wealth, previous, peak, cumulative_fee = 1.0, np.r_[1.0, np.zeros(len(market.panel.codes) - 1)], 1.0, 0.0
            for day in range(start_day, start_day + steps):
                target = _benchmark_target(method, day, market.panel, previous, index_weights)
                target = action_map(previous, target, market.panel.tradable[day], config.trade_fraction)
                old_wealth = wealth
                wealth, fee, turnover = next_wealth(wealth, previous, target, returns[day], config)
                cumulative_fee += float(fee)
                peak = max(peak, float(wealth))
                rows.append(dict(seed=int(seed), method=method, episode=(day - start_day) // config.horizon + 1,
                                 day=day, date=str(market.panel.dates[day]), regime=market.regime(day),
                                 wealth=float(wealth), reference=np.nan,
                                 net_return=float(wealth / old_wealth - 1.0),
                                 drawdown=float(1.0 - wealth / peak), cash=float(target[0]),
                                 turnover=float(turnover), fee=float(fee), cumulative_fee=cumulative_fee))
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
                         cumulative_fee=float(frame.fee.sum()), mean_cash=float(frame.cash.mean())))
    return pd.DataFrame(rows)


def _band(frame, value):
    grouped = frame.groupby("date")[value]
    return grouped.median(), grouped.quantile(.25), grouped.quantile(.75)


def plot_e5(daily, summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.3), constrained_layout=True)
    for method in [m for m in E5_METHODS if m in set(daily.method)]:
        frame = daily[daily.method == method]
        median, lower, upper = _band(frame, "wealth")
        x = pd.to_datetime(median.index)
        axes[0, 0].plot(x, median, label=METHOD_LABELS[method], color=PLOT_COLORS[method], lw=1.8)
        axes[0, 0].fill_between(x, lower.to_numpy(), upper.to_numpy(), color=PLOT_COLORS[method], alpha=.12)
        median, lower, upper = _band(frame, "drawdown")
        axes[0, 1].plot(x, median, label=METHOD_LABELS[method], color=PLOT_COLORS[method], lw=1.5)
        axes[0, 1].fill_between(x, lower.to_numpy(), upper.to_numpy(), color=PLOT_COLORS[method], alpha=.10)
    axes[0, 0].set_title("Out-of-sample wealth")
    axes[0, 1].set_title("Drawdown")
    axes[0, 0].set_ylabel("Wealth")
    axes[0, 1].set_ylabel("Drawdown")
    aggregate = summary.groupby("method", sort=False).median(numeric_only=True).reset_index()
    for _, row in aggregate.iterrows():
        method = row.method
        axes[1, 0].scatter(row.annualized_volatility, row.annualized_return,
                           color=PLOT_COLORS.get(method, "#333333"), s=48)
        axes[1, 0].annotate(METHOD_LABELS.get(method, method),
                             (row.annualized_volatility, row.annualized_return), fontsize=7,
                             xytext=(4, 3), textcoords="offset points")
    axes[1, 0].set_title("Return-risk comparison")
    axes[1, 0].set_xlabel("Annualized volatility")
    axes[1, 0].set_ylabel("Annualized return")
    for method in [m for m in E5_METHODS if m in set(daily.method)]:
        frame = daily[daily.method == method].copy()
        frame["cum_turnover"] = frame.groupby("seed").turnover.cumsum()
        median, lower, upper = _band(frame, "cum_turnover")
        x = pd.to_datetime(median.index)
        axes[1, 1].plot(x, median, label=METHOD_LABELS[method], color=PLOT_COLORS[method], lw=1.4)
        axes[1, 1].fill_between(x, lower.to_numpy(), upper.to_numpy(), color=PLOT_COLORS[method], alpha=.08)
    axes[1, 1].set_title("Cumulative one-sided turnover")
    axes[1, 1].set_ylabel("Turnover")
    for ax in axes.flat:
        ax.grid(alpha=.22, linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(output / "figure_E5_external_validity.pdf", bbox_inches="tight")
    fig.savefig(output / "figure_E5_external_validity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def prepare_real_panel(cache_root, panel_path, start, end, assets, pool_seed, allow_suspension_carry):
    metadata = prepare_panel(cache_root, panel_path, start, end, assets, pool_seed, allow_suspension_carry)
    return metadata


def run_e5(cache_root, panel_path, output, start_date="20230103", test_start="20250102",
           end_date="20260529", assets=300, pool_seed=2022, seeds=(29, 147, 3141),
           horizon=5, gamma=.05, trajectory_budget=256, evaluation_n=512,
           evaluation_m=256, eta_gain=.2, eta_loss=.05, cost=.001,
           trade_fraction=.2, allow_suspension_carry=True):
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
    config = PaperConfig(horizon=horizon, gamma=gamma, eta_gain=eta_gain, eta_loss=eta_loss,
                         cost=cost, trajectory_budget=trajectory_budget,
                         evaluation_n=evaluation_n, evaluation_m=evaluation_m,
                         dimension=min(10, panel.features.shape[-1]), trade_fraction=trade_fraction)
    available = len(panel.dates) - start_day
    steps = available - available % horizon
    if start_day <= horizon or steps < horizon:
        raise ValueError("Real-data test period is too short for the configured horizon")
    learned, learning_daily = run_online(market, config, list(seeds),
                                         ["dynamic", "symmetric", "static", "expected", "exponential"],
                                         steps, start_day, estimator="reuse")
    index_weights = load_hs300_weights(cache_root, panel.codes, panel.dates)
    benchmark_daily = run_benchmarks(market, config, list(seeds),
                                     ["equal_weight", "hs300_index", "inverse_volatility"],
                                     start_day, steps, index_weights)
    daily = pd.concat([learning_daily, benchmark_daily], ignore_index=True)
    summary = summarize_external(daily)
    learned.to_csv(output / "episodes.csv", index=False)
    daily.to_csv(output / "daily.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    coverage.to_csv(output / "quality_coverage.csv", index=False)
    pd.DataFrame([dict(date=str(date), source_date=source, available=weights is not None)
                  for date, (source, weights) in zip(panel.dates, index_weights)]).to_csv(
                      output / "index_weight_sources.csv", index=False)
    manifest = dict(experiment=5, status="complete", panel=str(panel_path), panel_sha256=file_hash(panel_path),
                    cache_root=str(cache_root), start_date=start_date, test_start=test_start,
                    end_date=end_date, assets=assets, pool_seed=pool_seed, seeds=list(seeds),
                    config=asdict(config), methods=E5_METHODS,
                    factor_timing="all features use information strictly before the execution day; announced fundamentals use ann_date < date",
                    return_basis="close/pre_close quote-relative; not a dividend-adjusted total-return series",
                    transaction_cost="c * ||target_t - target_{t-1}||_1, charged in next_wealth",
                    benchmark_index="latest cached HS300 snapshot on or before each day; missing assets renormalized",
                    dlr_warning="DLR is not used to rank real-market methods; this is an external-validity backtest")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_e5(daily, summary, output / "figures")
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


def _processed_opportunities(behavior_root, output, max_opportunities=0, epsilon=1e-6):
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
                    gain_eta = .2 if price >= current["asymmetric_ref"] else .05
                    current["symmetric_ref"] += .125 * (price - current["symmetric_ref"])
                    current["asymmetric_ref"] += gain_eta * (price - current["asymmetric_ref"])
                current["last_date"] = date
                if target_weight <= epsilon:
                    state.pop(code, None)
            if max_opportunities and len(records) >= max_opportunities:
                break
    if not records:
        raise ValueError("No eligible opportunities in processed 6purchase archive")
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    metadata = dict(rows=len(frame), source=str(archive), processed_layer="6purchase",
                    label="target weight decreases among stocks held immediately before the rebalance",
                    purchase_cost="stock_pool_beforexrxdpurchase after corporate-action adjustment",
                    max_opportunities=max_opportunities,
                    warning="PGR/PLR aggregate columns are not used as labels; y is reconstructed from target weights")
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def build_e6_opportunities(behavior_root, output, max_opportunities=0, epsilon=1e-6):
    """Reconstruct held-stock reduce opportunities from target-weight records.

    A row is produced only when the stock was held immediately before the
    rebalance and its target weight decreases.  Purchase-cost references are
    updated after the label is formed, so the label cannot leak the current
    decision into its own reference.
    """
    behavior_root, output = Path(behavior_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    processed = _processed_opportunities(behavior_root, output, max_opportunities, epsilon)
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
                current["last_date"] = date
            if target_weight <= epsilon:
                state.pop(code, None)
        if max_opportunities and len(records) >= max_opportunities:
            break
    if not records:
        raise ValueError("No held-stock opportunities were reconstructed")
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    metadata = dict(rows=len(frame), source_files=[str(p) for p in sorted((behavior_root / "raw_data" / "trading_data").glob("tm_trading*.xlsx"))],
                    corporate_action_files=[str(behavior_root / "xrxd1.xls"), str(behavior_root / "xrxd2.xls")],
                    label="target weight decreases among stocks held immediately before the rebalance",
                    epsilon=epsilon, max_opportunities=max_opportunities,
                    warning="PGR/PLR are diagnostic covariates; the y label is reconstructed from target weights, not copied from aggregate behavior tables")
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def _add_behavior_controls(frame, reference):
    x = frame.copy()
    x["log_age"] = np.log1p(np.maximum(x.age_days, 0))
    x["log_event_count"] = np.log1p(np.maximum(x.event_count, 0))
    x["log_prev_weight"] = np.log1p(np.maximum(x.prev_weight, 0))
    x["proactive"] = x.proactive.astype(float)
    controls = ["log_age", "log_event_count", "log_prev_weight", "proactive"]
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
    values = x[controls].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
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


def fit_e6_models(opportunities, output, bootstrap=200, seed=2026):
    frame = opportunities.copy()
    frame["date"] = pd.to_datetime(frame.date)
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
            model = LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)
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
                          "asymmetric_relative"]].copy()
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
    rng = np.random.default_rng(seed)
    user_ids = test.user_id.astype(str).unique()
    boot_rows = []
    for reference in ("none", "static", "symmetric", "asymmetric"):
        pred = predictions[predictions.reference == reference]
        for b in range(int(bootstrap)):
            sampled = rng.choice(user_ids, size=len(user_ids), replace=True)
            pieces = [pred[pred.user_id.astype(str) == user] for user in sampled]
            sample = pd.concat(pieces, ignore_index=True) if pieces else pred.iloc[:0]
            if sample.empty:
                continue
            met = _classification_metrics(sample.y, sample.predicted_reduce)
            boot_rows.append(dict(reference=reference, bootstrap=b, **met))
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
    behavior_rows = []
    relative_column = {"none": "static_relative", "static": "static_relative",
                       "symmetric": "symmetric_relative", "asymmetric": "asymmetric_relative"}
    for reference, group in predictions.groupby("reference"):
        relative = group[relative_column[reference]].to_numpy(float)
        observed = group.y.to_numpy(float)
        predicted = group.predicted_reduce.to_numpy(float)
        gain = relative >= 0
        for side, mask in (("PGR", gain), ("PLR", ~gain)):
            behavior_rows.append(dict(reference=reference, measure=side,
                                      observed=float(observed[mask].mean()) if mask.any() else np.nan,
                                      predicted=float(predicted[mask].mean()) if mask.any() else np.nan,
                                      n=int(mask.sum())))
        pgr = observed[gain].mean() if gain.any() else np.nan
        plr = observed[~gain].mean() if (~gain).any() else np.nan
        ppgr = predicted[gain].mean() if gain.any() else np.nan
        pplr = predicted[~gain].mean() if (~gain).any() else np.nan
        behavior_rows.append(dict(reference=reference, measure="PGR-PLR",
                                  observed=pgr - plr if np.isfinite(pgr) and np.isfinite(plr) else np.nan,
                                  predicted=ppgr - pplr if np.isfinite(ppgr) and np.isfinite(pplr) else np.nan,
                                  n=len(group)))
    pd.DataFrame(behavior_rows).to_csv(output / "pgr_plr_summary.csv", index=False)
    manifest = dict(experiment=6, status="complete", rows=len(frame), train_end=str(train_cut.date()),
                    validation_end=str(valid_cut.date()), test_rows=len(test), references=["none", "static", "symmetric", "asymmetric"],
                    label="held-stock target weight decrease at the next rebalance",
                    interpretation="supports behavioral alignment only if asymmetric improves out-of-sample calibration and PGR/PLR-related subgroup errors over all controls; no causal claim")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics, predictions


def plot_e6(metrics, predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    test = metrics[metrics.split == "test"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    order = ["none", "static", "symmetric", "asymmetric"]
    x = np.arange(len(order))
    for metric, color, label in (("log_loss", "#0072B2", "Log loss"),
                                 ("brier", "#D55E00", "Brier"),
                                 ("auprc", "#009E73", "AUPRC")):
        values = [test.loc[test.reference == ref, metric].iloc[0] if not test.loc[test.reference == ref, metric].empty else np.nan
                  for ref in order]
        axes[0].plot(x, values, marker="o", color=color, label=label)
    axes[0].set_xticks(x, order)
    axes[0].set_title("Out-of-sample estimator fit")
    axes[0].legend(frameon=False, fontsize=8)
    for ref, color in zip(order, ("#666666", "#009E73", "#D55E00", "#0072B2")):
        group = predictions[predictions.reference == ref]
        bins = pd.qcut(group.predicted_reduce.rank(method="first"), 10, labels=False, duplicates="drop")
        calib = group.assign(bin=bins).groupby("bin").agg(pred=("predicted_reduce", "mean"), obs=("y", "mean"))
        axes[1].plot(calib.pred, calib.obs, marker="o", ms=3, label=ref, color=color)
    axes[1].plot([0, 1], [0, 1], ls="--", color="#999999", lw=1)
    axes[1].set_xlabel("Predicted reduce probability")
    axes[1].set_ylabel("Observed reduce rate")
    axes[1].set_title("Calibration")
    behavior = pd.read_csv(output.parent / "pgr_plr_summary.csv") if (output.parent / "pgr_plr_summary.csv").exists() else None
    if behavior is not None:
        pivot = behavior.pivot(index="measure", columns="reference", values="observed").reindex(columns=order)
        im = axes[2].imshow(pivot.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
        axes[2].set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
        axes[2].set_yticks(np.arange(len(pivot.index)), pivot.index)
        axes[2].set_title("Observed PGR / PLR")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                if np.isfinite(value):
                    axes[2].text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axes[2], fraction=.046, pad=.04)
    for ax in axes:
        ax.grid(alpha=.18, linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "figure_E6_behavior_alignment.pdf", bbox_inches="tight")
    fig.savefig(output / "figure_E6_behavior_alignment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_agent_types(behavior_root, output, clusters=4, seed=2026):
    behavior_root, output = Path(behavior_root), Path(output)
    source = pd.read_pickle(behavior_root / "regression_afterholiday.pkl")
    source["date"] = pd.to_datetime(source["day"], errors="coerce")
    train_cut = source.date.dropna().sort_values().iloc[int(.70 * source.date.notna().sum())]
    train = source[source.date <= train_cut].copy()
    stats_columns = ["PGR", "PLR", "disposition", "transnum", "stknum", "num_posts"]
    for col in stats_columns:
        train[col] = pd.to_numeric(train[col], errors="coerce")
    stats = train.groupby("zhuliren")[stats_columns].mean().fillna(0.0)
    stats.index = stats.index.map(_normalise_code)
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
        stats["following_count"] = [len(follows.get(int(user), follows.get(user, []))) if str(user).isdigit() else 0
                                     for user in stats.index]
    else:
        stats["following_count"] = 0
    numeric = stats.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(numeric) < clusters:
        raise ValueError(f"Only {len(numeric)} users available for {clusters} agent clusters")
    scaler = StandardScaler().fit(numeric)
    labels = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit_predict(scaler.transform(numeric))
    stats["cluster"] = labels
    # Cluster names are descriptive labels, not claims that a real user belongs
    # to a psychological category.
    cluster_order = stats.groupby("cluster").disposition.mean().sort_values().index.tolist()
    names = {cluster: f"type_{i + 1}" for i, cluster in enumerate(cluster_order)}
    stats["agent_type"] = stats.cluster.map(names)
    stats.reset_index(names="user_id").to_csv(output / "agent_assignments.csv", index=False)
    profile = stats.groupby("agent_type")[stats.columns.difference(["cluster", "agent_type"])].mean()
    profile.reset_index().to_csv(output / "agent_type_profiles.csv", index=False)
    holdout = source[source.date > train_cut].copy()
    for col in stats_columns:
        holdout[col] = pd.to_numeric(holdout[col], errors="coerce")
    holdout_stats = holdout.groupby("zhuliren")[stats_columns].mean().fillna(0.0)
    holdout_stats["user_id"] = holdout_stats.index.map(_normalise_code)
    holdout_stats["agent_type"] = holdout_stats.user_id.map(stats.agent_type.to_dict())
    holdout_stats = holdout_stats.dropna(subset=["agent_type"])
    validation_rows = []
    for agent_type, group in holdout_stats.groupby("agent_type"):
        for column in stats_columns:
            train_value = float(profile.loc[agent_type, column]) if column in profile else np.nan
            observed = float(group[column].mean())
            validation_rows.append(dict(agent_type=agent_type, feature=column,
                                        train_profile=train_value, holdout_observed=observed,
                                        absolute_gap=abs(observed - train_value)))
    pd.DataFrame(validation_rows).to_csv(output / "agent_type_holdout_validation.csv", index=False)
    manifest = dict(source="behavior/regression_afterholiday.pkl + profile/network tables", clusters=clusters,
                    train_cut=str(train_cut.date()), social_timestamp_available=False,
                    holdout_validation="agent types are assigned from the training period and compared with held-out aggregate behavior",
                    warning="Agent types are calibrated simulation strata; no historical adoption or satisfaction labels are claimed")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stats, profile


def simulate_agents(e5_daily, assignments, profiles, output, seed=2026):
    output = Path(output)
    daily = e5_daily.copy()
    daily["date"] = pd.to_datetime(daily.date)
    market = (daily.groupby(["method", "date"], as_index=False)
              .agg(recommendation=("cash", lambda x: 1.0 - float(x.median())),
                   net_return=("net_return", "median"), turnover=("turnover", "median")))
    profile = profiles.copy()
    if "agent_type" in profile.columns:
        profile = profile.set_index("agent_type")
    # Empirical ranks make the simulator invariant to the scales in the source tables.
    for col in ("stknum", "transnum", "num_posts", "PGR", "PLR"):
        if col not in profile:
            profile[col] = 0.0
    def rank(values):
        return pd.Series(values).rank(pct=True).to_numpy()
    profile["stock_rank"] = rank(profile.stknum)
    profile["activity_rank"] = rank(profile.transnum)
    profile["risk_preference"] = np.clip(.15 + .65 * profile.stock_rank +
                                          .15 * profile.activity_rank +
                                          .05 * (profile.PGR - profile.PLR + 1.0), .1, .9)
    profile["loss_aversion"] = np.clip(1.5 + 1.5 * (profile.PLR - profile.PGR + .5), 1.0, 4.0)
    rng = np.random.default_rng(seed)
    rows = []
    for _, agent in profile.iterrows():
        agent_type = agent.name
        for _, row in market.iterrows():
            distance = abs(float(row.recommendation) - float(agent.risk_preference))
            invested_return = float(row.net_return) * float(agent.risk_preference)
            utility = (invested_return ** .88 if invested_return >= 0 else
                       -float(agent.loss_aversion) * (-invested_return) ** .88)
            satisfaction = utility - .001 * float(row.turnover) - .01 * distance
            satisfaction_bps = satisfaction * 10000.0
            adoption_probability = float(expit(-.7 + 1.8 * (1.0 - distance) +
                                               .8 * satisfaction_bps / 100.0 +
                                               .4 * float(agent.activity_rank)))
            rows.append(dict(agent_type=agent_type, method=row.method, date=row.date,
                             recommendation=float(row.recommendation), preferred_risky_weight=float(agent.risk_preference),
                             distance=distance, utility=float(utility), satisfaction=float(satisfaction),
                             satisfaction_bps=float(satisfaction_bps), adoption_probability=adoption_probability,
                             adopted=int(rng.random() < adoption_probability),
                             net_return=float(row.net_return), turnover=float(row.turnover)))
    traces = pd.DataFrame(rows)
    summary = (traces.groupby(["agent_type", "method"], as_index=False)
               .agg(adoption_rate=("adopted", "mean"), mean_adoption_probability=("adoption_probability", "mean"),
                    mean_satisfaction_bps=("satisfaction_bps", "mean"), cumulative_satisfaction=("satisfaction", "sum"),
                    mean_distance=("distance", "mean")))
    traces.to_csv(output / "agent_traces.csv", index=False)
    summary.to_csv(output / "agent_method_summary.csv", index=False)
    profile.reset_index().to_csv(output / "agent_type_profiles_simulated.csv", index=False)
    manifest = dict(experiment=7, status="complete", methods=sorted(market.method.unique()),
                    loop="market observation -> method recommendation -> calibrated agent utility -> simulated adoption -> next observation",
                    adoption="simulated Bernoulli draw from an explicit compatibility/utility rule",
                    satisfaction="agent-specific simulated CPT-like utility net of turnover and recommendation mismatch",
                    limitation="no real adoption or satisfaction label is available; results are a calibrated extension, not a user study")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary, traces


def plot_e7(summary, profiles, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    order_types = sorted(summary.agent_type.unique())
    order_methods = [m for m in E5_METHODS if m in set(summary.method)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    profile_cols = [c for c in ("PGR", "PLR", "disposition", "transnum", "stknum") if c in profiles]
    profile_frame = profiles.set_index("agent_type").reindex(order_types)[profile_cols]
    im = axes[0, 0].imshow(StandardScaler().fit_transform(profile_frame), aspect="auto", cmap="RdBu_r")
    axes[0, 0].set_xticks(np.arange(len(profile_cols)), profile_cols, rotation=35, ha="right")
    axes[0, 0].set_yticks(np.arange(len(order_types)), order_types)
    axes[0, 0].set_title("Calibrated agent strata")
    fig.colorbar(im, ax=axes[0, 0], fraction=.046, pad=.04)
    for ax, value, title in ((axes[0, 1], "adoption_rate", "Simulated adoption rate"),
                             (axes[1, 0], "mean_satisfaction_bps", "Simulated satisfaction (bps)")):
        matrix = summary.pivot(index="agent_type", columns="method", values=value).reindex(index=order_types, columns=order_methods)
        im = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="YlGn")
        ax.set_xticks(np.arange(len(order_methods)), [METHOD_LABELS[m] for m in order_methods], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(order_types)), order_types)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix.iloc[i, j]):
                    ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    for agent_type, group in summary.groupby("agent_type"):
        axes[1, 1].scatter(group.mean_adoption_probability, group.mean_satisfaction_bps,
                           label=agent_type, s=45)
    axes[1, 1].set_xlabel("Mean adoption probability")
    axes[1, 1].set_ylabel("Mean satisfaction (bps)")
    axes[1, 1].set_title("Adoption--satisfaction relation")
    axes[1, 1].legend(frameon=False, fontsize=7)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15, linewidth=.5)
    fig.savefig(output / "figure_E7_heterogeneous_agents.pdf", bbox_inches="tight")
    fig.savefig(output / "figure_E7_heterogeneous_agents.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
