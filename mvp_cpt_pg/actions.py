from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .market_data import factorize_state

CASH_CODE = "__CASH__"
MIN_ACTIVE_WEIGHT = 1e-12
POLICY_FEATURE_BOUND = 10.0

POLICY_FEATURE_COLUMNS = [
    "momentum_score",
    "value_score",
    "quality_score",
    "low_vol_score",
    "mean_reversion_score",
    "qbot_boll_reversion_score",
    "qbot_rsi_reversal_score",
    "qbot_macd_trend_score",
    "qbot_rsrs_timing_score",
    "balanced_score",
]

@dataclass(frozen=True)
class ContinuousPolicyState:
    codes: list[str]
    feature_matrix: np.ndarray
    frame: pd.DataFrame


def build_continuous_policy_state(
    state: pd.DataFrame,
) -> ContinuousPolicyState:
    frame = factorize_state(state).sort_values("ts_code").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("Observed market state is empty")
    if CASH_CODE in set(frame["ts_code"].astype(str)):
        raise RuntimeError(f"Market state contains reserved cash code: {CASH_CODE}")
    cash_row = {column: 0.0 for column in frame.columns}
    cash_row["ts_code"] = CASH_CODE
    frame = pd.concat([pd.DataFrame([cash_row]), frame], ignore_index=True)
    feature_matrix = _build_contextual_feature_matrix(frame)
    return ContinuousPolicyState(
        codes=frame["ts_code"].tolist(),
        feature_matrix=feature_matrix,
        frame=frame,
    )


def continuous_action_summary(
    weights: pd.Series,
    prev_weights: pd.Series,
    portfolio_value: float,
    display_threshold: float | None = None,
) -> dict[str, Any]:
    threshold = MIN_ACTIVE_WEIGHT if display_threshold is None else float(display_threshold)
    holding_count = effective_holding_count(weights, threshold=threshold)
    trades: list[dict[str, Any]] = []
    all_codes = sorted(set(prev_weights.index).union(weights.index))
    prev = prev_weights.reindex(all_codes, fill_value=0.0)
    new = weights.reindex(all_codes, fill_value=0.0)
    changes = (new - prev).sort_values(key=lambda s: s.abs(), ascending=False)
    buy_amount = 0.0
    sell_amount = 0.0
    for code, delta_weight in changes.items():
        if abs(float(delta_weight)) < 1e-8:
            continue
        amount = float(delta_weight) * float(portfolio_value)
        direction = "buy" if amount >= 0 else "sell"
        if code == CASH_CODE:
            direction = "cash_increase" if amount >= 0 else "cash_decrease"
        elif amount >= 0:
            buy_amount += amount
        else:
            sell_amount += abs(amount)
        trades.append(
            {
                "code": code,
                "direction": direction,
                "amount": round(abs(amount), 2),
                "from_weight": round(float(prev.loc[code]), 6),
                "to_weight": round(float(new.loc[code]), 6),
                "delta_weight": round(float(delta_weight), 6),
            }
        )
    return {
        "name": "continuous_weight_policy",
        "portfolio_weights": portfolio_weights_dict(weights, threshold=threshold),
        "holding_count": holding_count,
        "trade_plan": trades,
        "buy_amount": round(buy_amount, 2),
        "sell_amount": round(sell_amount, 2),
        "cash_after_trade": round(float(weights.get(CASH_CODE, 0.0)), 6),
    }


def effective_risky_weights(weights: pd.Series, threshold: float = MIN_ACTIVE_WEIGHT) -> pd.Series:
    risky_weights = weights.drop(labels=[CASH_CODE], errors="ignore").clip(lower=0.0).astype(float)
    return risky_weights[risky_weights >= threshold]


def effective_holding_count(weights: pd.Series, threshold: float = MIN_ACTIVE_WEIGHT) -> int:
    return int(len(effective_risky_weights(weights, threshold=threshold)))


def portfolio_weights_dict(weights: pd.Series, threshold: float = MIN_ACTIVE_WEIGHT) -> dict[str, float]:
    display_weights = effective_risky_weights(weights, threshold=threshold).sort_values(ascending=False)
    output = {code: round(float(weight), 6) for code, weight in display_weights.items()}
    cash_weight = float(weights.get(CASH_CODE, 0.0))
    if cash_weight > 1e-8 or CASH_CODE in weights.index:
        output[CASH_CODE] = round(cash_weight, 6)
    return output


def portfolio_weights_percent_dict(weights: pd.Series, threshold: float = MIN_ACTIVE_WEIGHT) -> dict[str, float]:
    return {code: round(float(weight) * 100.0, 4) for code, weight in portfolio_weights_dict(weights, threshold=threshold).items()}


def format_portfolio_weights_percent(weights: pd.Series, threshold: float = MIN_ACTIVE_WEIGHT) -> str:
    return json.dumps(portfolio_weights_percent_dict(weights, threshold=threshold), ensure_ascii=False, sort_keys=True)


def _build_contextual_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    feature_matrix = frame[POLICY_FEATURE_COLUMNS].to_numpy(dtype=float)
    return np.clip(feature_matrix, -POLICY_FEATURE_BOUND, POLICY_FEATURE_BOUND)

