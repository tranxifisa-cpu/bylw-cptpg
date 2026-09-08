from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


TAIL_FRACTION = 0.20
DYNAMIC_LOCAL_REGRET_ALPHA = 0.90


def smoothed_gradient(
    history: list[np.ndarray],
    gradient: np.ndarray,
    *,
    window: int,
    alpha: float = DYNAMIC_LOCAL_REGRET_ALPHA,
) -> np.ndarray:
    if window <= 0:
        raise ValueError("dynamic local regret window must be positive")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("dynamic local regret alpha must be in (0, 1]")
    gradient = np.asarray(gradient, dtype=float)
    if gradient.ndim != 1:
        raise ValueError("dynamic local regret gradient must be a one-dimensional vector")
    vectors = [gradient, *list(reversed(history[-max(0, window - 1) :]))]
    if not vectors:
        return float("nan")
    weights = np.asarray([alpha**i for i in range(len(vectors))], dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return float("nan")
    smoothed = np.zeros_like(gradient, dtype=float)
    for weight, vector in zip(weights, vectors):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != gradient.shape:
            raise ValueError("dynamic local regret gradient history has incompatible shape")
        smoothed += weight * vector
    smoothed /= total_weight
    return smoothed


def dynamic_local_regret_value(
    history: list[np.ndarray],
    gradient: np.ndarray,
    *,
    window: int,
    alpha: float = DYNAMIC_LOCAL_REGRET_ALPHA,
) -> float:
    smoothed = smoothed_gradient(history, gradient, window=window, alpha=alpha)
    return float(np.dot(smoothed, smoothed))


def add_dynamic_local_regret_columns(
    trace: pd.DataFrame,
    *,
    window: int,
    alpha: float = DYNAMIC_LOCAL_REGRET_ALPHA,
) -> pd.DataFrame:
    if trace.empty:
        return trace.copy()
    if "dynamic_local_regret" in trace.columns and "dynamic_local_regret_term" in trace.columns:
        return trace.copy()
    data = trace.copy()
    group_columns = ["run_key"] if "run_key" in data.columns else ["method", "seed"]
    sort_column = "trade_date" if "trade_date" in data.columns else "step"
    term_values = pd.Series(index=data.index, dtype=float)
    cumulative_values = pd.Series(index=data.index, dtype=float)
    average_values = pd.Series(index=data.index, dtype=float)
    for _, frame in data.groupby(group_columns, sort=False):
        history: list[np.ndarray] = []
        cumulative = 0.0
        regret_count = 0
        ordered = frame.sort_values(sort_column) if sort_column in frame.columns else frame
        on_policy_episode = (
            "episode_update" in ordered.columns
            and "estimation_mode" in ordered.columns
            and (ordered["estimation_mode"] == "on_policy_episode").any()
        )
        for index, row in ordered.iterrows():
            if on_policy_episode and int(pd.to_numeric(row.get("episode_update", 0), errors="coerce") or 0) != 1:
                cumulative_values.at[index] = cumulative
                average_values.at[index] = cumulative / regret_count if regret_count > 0 else math.nan
                continue
            gradient = _row_gradient_vector(row)
            if history and history[-1].shape != gradient.shape:
                history = []
                cumulative = 0.0
                regret_count = 0
            value = dynamic_local_regret_value(history, gradient, window=window, alpha=alpha)
            history.append(gradient)
            cumulative += value
            regret_count += 1
            term_values.at[index] = value
            cumulative_values.at[index] = cumulative
            average_values.at[index] = cumulative / regret_count
    data["dynamic_local_regret_term"] = term_values
    data["dynamic_local_regret"] = cumulative_values
    data["average_dynamic_local_regret"] = average_values
    return data


def _row_gradient_vector(row: pd.Series) -> np.ndarray:
    if "gradient_vector" in row.index:
        raw = row["gradient_vector"]
        if isinstance(raw, str) and raw.strip():
            try:
                values = json.loads(raw)
            except json.JSONDecodeError:
                values = []
            if isinstance(values, list) and values:
                return np.asarray(values, dtype=float)
        if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
            return np.asarray(raw, dtype=float)
    norm = float(pd.to_numeric(row.get("gradient_norm", 0.0), errors="coerce"))
    if not math.isfinite(norm):
        norm = 0.0
    return np.asarray([norm], dtype=float)


def max_drawdown(values: pd.Series) -> float:
    wealth = pd.to_numeric(values, errors="coerce").ffill().fillna(1.0)
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series) -> float:
    series = pd.to_numeric(returns, errors="coerce").dropna()
    if len(series) < 2:
        return 0.0
    std = float(series.std(ddof=0))
    if std < 1e-12:
        return 0.0
    return float(series.mean() / std * math.sqrt(252))


def summarize_runs(trace: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "seed",
        "completed_days",
        "final_wealth",
        "sharpe",
        "max_drawdown",
        "turnover_mean",
        "holding_count_mean",
        "objective_estimate_mean",
        "offline_cpt_common_ref_mean",
        "gradient_norm_mean",
        "dynamic_local_regret_term_mean",
        "average_dynamic_local_regret",
        "dynamic_local_regret",
        "average_squared_gradient_norm",
        "cumulative_squared_gradient_norm",
        "projected_gradient_mapping_norm_mean",
        "average_squared_pgm_norm",
        "cumulative_squared_pgm_norm",
        "gradient_bootstrap_error_norm_mean",
        "gradient_bootstrap_error_tail20_mean",
        "gradient_bootstrap_std_norm_mean",
        "gradient_bootstrap_se_norm_mean",
        "gradient_bootstrap_relative_error_mean",
        "objective_bootstrap_std_mean",
        "gradient_diagnostic_repeats_mean",
        "cpt_sample_count_mean",
        "gradient_sample_count_mean",
        "theta_norm_mean",
        "theta_max_abs_mean",
        "theta_boundary_share_mean",
        "reference_drift_mean",
        "reference_path_variation",
    ]
    if trace.empty:
        return pd.DataFrame(columns=columns)
    records = []
    for (method, seed), frame in trace.groupby(["method", "seed"]):
        frame = frame.sort_values("trade_date")
        holding_count_mean = (
            float(pd.to_numeric(frame["holding_count"], errors="coerce").mean())
            if "holding_count" in frame.columns
            else float("nan")
        )
        offline_cpt_common_ref_mean = (
            float(pd.to_numeric(frame["offline_cpt_common_ref"], errors="coerce").mean())
            if "offline_cpt_common_ref" in frame.columns
            else float("nan")
        )
        gradient_norm = pd.to_numeric(frame["gradient_norm"], errors="coerce").dropna()
        squared_gradient_norm = gradient_norm**2
        dynamic_local_regret_term = (
            pd.to_numeric(frame["dynamic_local_regret_term"], errors="coerce")
            if "dynamic_local_regret_term" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        average_dynamic_local_regret = (
            pd.to_numeric(frame["average_dynamic_local_regret"], errors="coerce")
            if "average_dynamic_local_regret" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        dynamic_local_regret = (
            pd.to_numeric(frame["dynamic_local_regret"], errors="coerce")
            if "dynamic_local_regret" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        pgm_norm = (
            pd.to_numeric(frame["projected_gradient_mapping_norm"], errors="coerce")
            if "projected_gradient_mapping_norm" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        squared_pgm_norm = pgm_norm**2
        tail_count = max(1, int(math.ceil(len(frame) * TAIL_FRACTION)))
        gradient_bootstrap_error = (
            pd.to_numeric(frame["gradient_bootstrap_error_norm"], errors="coerce")
            if "gradient_bootstrap_error_norm" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        gradient_bootstrap_std = (
            pd.to_numeric(frame["gradient_bootstrap_std_norm"], errors="coerce")
            if "gradient_bootstrap_std_norm" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        gradient_bootstrap_se = (
            pd.to_numeric(frame["gradient_bootstrap_se_norm"], errors="coerce")
            if "gradient_bootstrap_se_norm" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        gradient_bootstrap_relative_error = (
            pd.to_numeric(frame["gradient_bootstrap_relative_error"], errors="coerce")
            if "gradient_bootstrap_relative_error" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        objective_bootstrap_std = (
            pd.to_numeric(frame["objective_bootstrap_std"], errors="coerce")
            if "objective_bootstrap_std" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        diagnostic_repeats = (
            pd.to_numeric(frame["gradient_diagnostic_repeats"], errors="coerce")
            if "gradient_diagnostic_repeats" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        cpt_sample_count = (
            pd.to_numeric(frame["cpt_sample_count"], errors="coerce")
            if "cpt_sample_count" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        gradient_sample_count = (
            pd.to_numeric(frame["gradient_sample_count"], errors="coerce")
            if "gradient_sample_count" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        theta_norm = (
            pd.to_numeric(frame["theta_norm"], errors="coerce")
            if "theta_norm" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        theta_max_abs = (
            pd.to_numeric(frame["theta_max_abs"], errors="coerce")
            if "theta_max_abs" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        theta_boundary_share = (
            pd.to_numeric(frame["theta_boundary_share"], errors="coerce")
            if "theta_boundary_share" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        gradient_bootstrap_error_tail = gradient_bootstrap_error.tail(tail_count)
        reference_drift = (
            pd.to_numeric(frame["reference_drift"], errors="coerce")
            if "reference_drift" in frame.columns
            else pd.Series(dtype=float)
        ).dropna()
        records.append(
            {
                "method": method,
                "seed": seed,
                "completed_days": int(len(frame)),
                "final_wealth": float(frame["wealth"].iloc[-1]),
                "sharpe": sharpe_ratio(frame["day_return_rate"]),
                "max_drawdown": max_drawdown(frame["wealth"]),
                "turnover_mean": float(frame["turnover"].mean()),
                "holding_count_mean": holding_count_mean,
                "objective_estimate_mean": float(frame["objective_estimate"].mean()),
                "offline_cpt_common_ref_mean": offline_cpt_common_ref_mean,
                "gradient_norm_mean": float(frame["gradient_norm"].mean()),
                "dynamic_local_regret_term_mean": (
                    float(dynamic_local_regret_term.mean()) if not dynamic_local_regret_term.empty else float("nan")
                ),
                "average_dynamic_local_regret": (
                    float(average_dynamic_local_regret.iloc[-1]) if not average_dynamic_local_regret.empty else float("nan")
                ),
                "dynamic_local_regret": (
                    float(dynamic_local_regret.iloc[-1]) if not dynamic_local_regret.empty else float("nan")
                ),
                "average_squared_gradient_norm": float(squared_gradient_norm.mean()) if not squared_gradient_norm.empty else float("nan"),
                "cumulative_squared_gradient_norm": float(squared_gradient_norm.sum()) if not squared_gradient_norm.empty else float("nan"),
                "projected_gradient_mapping_norm_mean": float(pgm_norm.mean()) if not pgm_norm.empty else float("nan"),
                "average_squared_pgm_norm": float(squared_pgm_norm.mean()) if not squared_pgm_norm.empty else float("nan"),
                "cumulative_squared_pgm_norm": float(squared_pgm_norm.sum()) if not squared_pgm_norm.empty else float("nan"),
                "gradient_bootstrap_error_norm_mean": float(gradient_bootstrap_error.mean()) if not gradient_bootstrap_error.empty else float("nan"),
                "gradient_bootstrap_error_tail20_mean": (
                    float(gradient_bootstrap_error_tail.mean()) if not gradient_bootstrap_error_tail.empty else float("nan")
                ),
                "gradient_bootstrap_std_norm_mean": float(gradient_bootstrap_std.mean()) if not gradient_bootstrap_std.empty else float("nan"),
                "gradient_bootstrap_se_norm_mean": float(gradient_bootstrap_se.mean()) if not gradient_bootstrap_se.empty else float("nan"),
                "gradient_bootstrap_relative_error_mean": (
                    float(gradient_bootstrap_relative_error.mean()) if not gradient_bootstrap_relative_error.empty else float("nan")
                ),
                "objective_bootstrap_std_mean": float(objective_bootstrap_std.mean()) if not objective_bootstrap_std.empty else float("nan"),
                "gradient_diagnostic_repeats_mean": float(diagnostic_repeats.mean()) if not diagnostic_repeats.empty else float("nan"),
                "cpt_sample_count_mean": float(cpt_sample_count.mean()) if not cpt_sample_count.empty else float("nan"),
                "gradient_sample_count_mean": float(gradient_sample_count.mean()) if not gradient_sample_count.empty else float("nan"),
                "theta_norm_mean": float(theta_norm.mean()) if not theta_norm.empty else float("nan"),
                "theta_max_abs_mean": float(theta_max_abs.mean()) if not theta_max_abs.empty else float("nan"),
                "theta_boundary_share_mean": float(theta_boundary_share.mean()) if not theta_boundary_share.empty else float("nan"),
                "reference_drift_mean": float(reference_drift.mean()) if not reference_drift.empty else float("nan"),
                "reference_path_variation": float(reference_drift.sum()) if not reference_drift.empty else float("nan"),
            }
        )
    return pd.DataFrame(records, columns=columns).sort_values(["method", "seed"]).reset_index(drop=True)


def aggregate_methods(summary: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "final_wealth",
        "sharpe",
        "max_drawdown",
        "turnover_mean",
        "holding_count_mean",
        "objective_estimate_mean",
        "offline_cpt_common_ref_mean",
        "gradient_norm_mean",
        "dynamic_local_regret_term_mean",
        "average_dynamic_local_regret",
        "dynamic_local_regret",
        "average_squared_gradient_norm",
        "cumulative_squared_gradient_norm",
        "projected_gradient_mapping_norm_mean",
        "average_squared_pgm_norm",
        "cumulative_squared_pgm_norm",
        "gradient_bootstrap_error_norm_mean",
        "gradient_bootstrap_error_tail20_mean",
        "gradient_bootstrap_std_norm_mean",
        "gradient_bootstrap_se_norm_mean",
        "gradient_bootstrap_relative_error_mean",
        "objective_bootstrap_std_mean",
        "gradient_diagnostic_repeats_mean",
        "cpt_sample_count_mean",
        "gradient_sample_count_mean",
        "theta_norm_mean",
        "theta_max_abs_mean",
        "theta_boundary_share_mean",
        "reference_drift_mean",
        "reference_path_variation",
    ]
    columns = [
        "method",
        "completed_days",
        "run_count",
        *[f"{name}_mean" for name in numeric_cols],
        *[f"{name}_std" for name in numeric_cols],
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    target_days = int(summary["completed_days"].max())
    filtered = summary[summary["completed_days"] == target_days].copy()
    grouped = filtered.groupby("method", as_index=False)
    mean_df = grouped[numeric_cols].mean().rename(columns={name: f"{name}_mean" for name in numeric_cols})
    std_df = grouped[numeric_cols].std(ddof=0).fillna(0.0).rename(columns={name: f"{name}_std" for name in numeric_cols})
    count_df = grouped.agg(completed_days=("completed_days", "first"), run_count=("seed", "count"))
    merged = count_df.merge(mean_df, on="method", how="left").merge(std_df, on="method", how="left")
    return merged[columns].sort_values("method").reset_index(drop=True)
