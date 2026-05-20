from __future__ import annotations

import math

import pandas as pd


ADOPTION_SCORE = {"adopt": 1.0, "partial": 0.5, "skip": 0.0}
TAIL_FRACTION = 0.20


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
        "rating_mean",
        "adoption_rate",
        "constraint_violation_rate",
        "constraint_violation_reason_counts",
        "turnover_mean",
        "holding_count_mean",
        "objective_estimate_mean",
        "offline_cpt_common_ref_mean",
        "gradient_norm_mean",
        "average_squared_gradient_norm",
        "cumulative_squared_gradient_norm",
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
        adoption_rate = frame["adoption"].map(ADOPTION_SCORE).fillna(0.0).mean()
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
        reason_counts = ""
        if "constraint_violation_reason" in frame.columns:
            reasons = frame["constraint_violation_reason"].fillna("").astype(str)
            reasons = reasons[reasons.ne("")]
            if not reasons.empty:
                reason_counts = ";".join(f"{reason}:{count}" for reason, count in reasons.value_counts().sort_index().items())
        records.append(
            {
                "method": method,
                "seed": seed,
                "completed_days": int(len(frame)),
                "final_wealth": float(frame["wealth"].iloc[-1]),
                "sharpe": sharpe_ratio(frame["day_return_rate"]),
                "max_drawdown": max_drawdown(frame["wealth"]),
                "rating_mean": float(pd.to_numeric(frame["rating"], errors="coerce").mean()),
                "adoption_rate": float(adoption_rate),
                "constraint_violation_rate": float(frame["constraint_violation"].mean()),
                "constraint_violation_reason_counts": reason_counts,
                "turnover_mean": float(frame["turnover"].mean()),
                "holding_count_mean": holding_count_mean,
                "objective_estimate_mean": float(frame["objective_estimate"].mean()),
                "offline_cpt_common_ref_mean": offline_cpt_common_ref_mean,
                "gradient_norm_mean": float(frame["gradient_norm"].mean()),
                "average_squared_gradient_norm": float(squared_gradient_norm.mean()) if not squared_gradient_norm.empty else float("nan"),
                "cumulative_squared_gradient_norm": float(squared_gradient_norm.sum()) if not squared_gradient_norm.empty else float("nan"),
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
        "rating_mean",
        "adoption_rate",
        "constraint_violation_rate",
        "turnover_mean",
        "holding_count_mean",
        "objective_estimate_mean",
        "offline_cpt_common_ref_mean",
        "gradient_norm_mean",
        "average_squared_gradient_norm",
        "cumulative_squared_gradient_norm",
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
