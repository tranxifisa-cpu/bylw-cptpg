from __future__ import annotations

import math
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .market_data import factorize_state
from .schemas import HardConstraints, MIN_SINGLE_STOCK_WEIGHT, PreferenceVector
from .utils import normalize_weights

CASH_CODE = "__CASH__"
MIN_ACTIVE_WEIGHT = MIN_SINGLE_STOCK_WEIGHT

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
    news_context: dict[str, float]


def build_news_context(state: pd.DataFrame) -> dict[str, float]:
    count = _frame_scalar(state, "news_total_count")
    positive = _frame_scalar(state, "news_total_positive")
    negative = _frame_scalar(state, "news_total_negative")
    raw_sentiment = positive - negative
    sentiment_scale = math.sqrt(max(count, 1.0))
    return {
        "news_total_count": count,
        "news_total_positive": positive,
        "news_total_negative": negative,
        "news_sentiment": raw_sentiment,
        "news_sentiment_score": math.tanh(raw_sentiment / sentiment_scale),
        "news_risk_pressure": negative / max(count, 1.0),
        "news_attention": math.log1p(max(count, 0.0)),
    }


def build_continuous_policy_state(
    state: pd.DataFrame,
    preference: PreferenceVector,
    prev_weights: pd.Series | None = None,
) -> ContinuousPolicyState:
    frame = factorize_state(state).sort_values("ts_code").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("Observed market state is empty")
    news_context = build_news_context(state)
    feature_matrix = _build_contextual_feature_matrix(frame, preference, news_context)
    return ContinuousPolicyState(
        codes=frame["ts_code"].tolist(),
        feature_matrix=feature_matrix,
        frame=frame,
        news_context=news_context,
    )
def project_continuous_weights(
    codes: list[str],
    raw_weights: np.ndarray | pd.Series,
    hard_constraints: HardConstraints,
    prev_weights: pd.Series,
) -> pd.Series:
    raw_array = np.asarray(raw_weights, dtype=float)
    prev_array = prev_weights.reindex(codes, fill_value=0.0).to_numpy(dtype=float)
    projected = project_continuous_weights_array(
        codes=codes,
        raw_weights=raw_array,
        hard_constraints=hard_constraints,
        prev_weights=prev_array,
    )
    return pd.Series(projected, index=codes, dtype=float)


def project_continuous_weights_array(
    codes: list[str],
    raw_weights: np.ndarray,
    hard_constraints: HardConstraints,
    prev_weights: np.ndarray,
) -> np.ndarray:
    if len(codes) != len(raw_weights) or len(codes) != len(prev_weights):
        raise RuntimeError("Projected continuous portfolio array shape mismatch")
    if CASH_CODE not in codes:
        raise RuntimeError("Cash asset is missing from policy action space")
    cash_index = codes.index(CASH_CODE)
    risky_indices = np.array([index for index, code in enumerate(codes) if code != CASH_CODE], dtype=int)
    max_weight = float(max(1e-6, min(1.0, hard_constraints.max_single_weight)))
    risk_budget = float(max(0.0, min(1.0, hard_constraints.risk_budget)))
    turnover_cap = float(max(0.0, min(2.0, hard_constraints.turnover_cap)))

    weights = np.maximum(np.asarray(raw_weights, dtype=float), 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("Projected continuous portfolio weights are non-positive")
    weights = weights / total
    risky = weights[risky_indices].copy()

    risky = np.maximum(risky, 0.0)
    for _ in range(8):
        over = risky > max_weight
        if not bool(over.any()):
            break
        excess = float((risky[over] - max_weight).sum())
        risky[over] = max_weight
        under = risky < max_weight - 1e-12
        if not bool(under.any()) or excess <= 1e-12:
            break
        room = max_weight - risky[under]
        room_sum = float(room.sum())
        if room_sum <= 1e-12:
            break
        risky[under] += excess * (room / room_sum)

    positive_indices = np.flatnonzero(risky > 0.0)
    if len(positive_indices) > 0:
        target_sum = float(risky[positive_indices].sum())
        feasible_count = int(target_sum // MIN_ACTIVE_WEIGHT)
        if feasible_count < 1 or target_sum <= 0.0:
            risky = np.zeros_like(risky)
        else:
            target_count = max(1, min(int(hard_constraints.diversification_target), len(positive_indices), feasible_count))
            target_sum = min(target_sum, float(target_count) * max_weight)
            selected_order = np.argsort(-risky[positive_indices], kind="mergesort")[:target_count]
            selected_indices = positive_indices[selected_order]
            selected = risky[selected_indices]
            selected_total = float(selected.sum())
            output = np.zeros_like(risky)
            if selected_total > 0.0 and target_sum > 0.0:
                selected = selected / selected_total * target_sum
                selected = np.maximum(selected, 0.0)
                for _ in range(8):
                    over = selected > max_weight
                    if not bool(over.any()):
                        break
                    excess = float((selected[over] - max_weight).sum())
                    selected[over] = max_weight
                    under = selected < max_weight - 1e-12
                    if not bool(under.any()) or excess <= 1e-12:
                        break
                    room = max_weight - selected[under]
                    room_sum = float(room.sum())
                    if room_sum <= 1e-12:
                        break
                    selected[under] += excess * (room / room_sum)
                output[selected_indices] = selected
            risky = output
    risky = np.where(risky >= MIN_ACTIVE_WEIGHT, risky, 0.0)
    risky_sum = float(risky.sum())
    if risky_sum > 1.0:
        risky = risky / risky_sum

    prev_risky = np.maximum(np.asarray(prev_weights, dtype=float)[risky_indices], 0.0)
    prev_risky = np.where(prev_risky >= MIN_ACTIVE_WEIGHT, prev_risky, 0.0)
    target_risky = np.where(risky >= MIN_ACTIVE_WEIGHT, risky, 0.0)
    turnover = float(np.abs(target_risky - prev_risky).sum())
    if turnover > turnover_cap + 1e-12 and turnover > 1e-12:
        mix = min(1.0, turnover_cap / turnover)
        risky = np.maximum(prev_risky + mix * (target_risky - prev_risky), 0.0)
    else:
        risky = target_risky

    positive_indices = np.flatnonzero(risky > 0.0)
    if len(positive_indices) > 0:
        target_sum = float(risky[positive_indices].sum())
        feasible_count = int(target_sum // MIN_ACTIVE_WEIGHT)
        if feasible_count < 1 or target_sum <= 0.0:
            risky = np.zeros_like(risky)
        else:
            target_count = max(1, min(int(hard_constraints.diversification_target), len(positive_indices), feasible_count))
            target_sum = min(target_sum, float(target_count) * max_weight)
            selected_order = np.argsort(-risky[positive_indices], kind="mergesort")[:target_count]
            selected_indices = positive_indices[selected_order]
            selected = risky[selected_indices]
            selected_total = float(selected.sum())
            output = np.zeros_like(risky)
            if selected_total > 0.0 and target_sum > 0.0:
                selected = selected / selected_total * target_sum
                selected = np.maximum(selected, 0.0)
                for _ in range(8):
                    over = selected > max_weight
                    if not bool(over.any()):
                        break
                    excess = float((selected[over] - max_weight).sum())
                    selected[over] = max_weight
                    under = selected < max_weight - 1e-12
                    if not bool(under.any()) or excess <= 1e-12:
                        break
                    room = max_weight - selected[under]
                    room_sum = float(room.sum())
                    if room_sum <= 1e-12:
                        break
                    selected[under] += excess * (room / room_sum)
                output[selected_indices] = selected
            risky = output
    risky = np.where((risky >= MIN_ACTIVE_WEIGHT) | (prev_risky >= MIN_ACTIVE_WEIGHT), risky, 0.0)
    prev_cash_weight = max(float(np.asarray(prev_weights, dtype=float)[cash_index]), 0.0)
    min_cash_weight = max(0.0, prev_cash_weight * (1.0 - risk_budget))
    max_risky_sum = min(1.0, 1.0 - min_cash_weight)
    risky_sum = float(risky.sum())
    if risky_sum > max_risky_sum and risky_sum > 1e-12:
        risky = risky * (max_risky_sum / risky_sum)
        risky = np.where((risky >= MIN_ACTIVE_WEIGHT) | (prev_risky >= MIN_ACTIVE_WEIGHT), risky, 0.0)

    output = np.zeros(len(codes), dtype=float)
    output[risky_indices] = np.maximum(risky, 0.0)
    output[cash_index] = max(0.0, 1.0 - float(output[risky_indices].sum()))
    total = float(output.sum())
    if total <= 0.0 or output[cash_index] < -1e-8:
        raise RuntimeError("Projected continuous portfolio weights are non-positive")
    return output / total


def continuous_action_summary(
    weights: pd.Series,
    prev_weights: pd.Series,
    portfolio_value: float,
    news_context: dict[str, float],
) -> dict[str, Any]:
    holding_count = effective_holding_count(weights)
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
        "portfolio_weights": portfolio_weights_dict(weights),
        "holding_count": holding_count,
        "trade_plan": trades,
        "buy_amount": round(buy_amount, 2),
        "sell_amount": round(sell_amount, 2),
        "cash_after_trade": round(float(weights.get(CASH_CODE, 0.0)), 6),
        "news_adjustment": float(news_context["news_sentiment_score"]),
    }


def check_constraint_violation(
    weights: pd.Series,
    hard_constraints: HardConstraints,
    prev_weights: pd.Series | None = None,
) -> int:
    return 0 if constraint_violation_reason(weights, hard_constraints, prev_weights) == "" else 1


def constraint_violation_reason(
    weights: pd.Series,
    hard_constraints: HardConstraints,
    prev_weights: pd.Series | None = None,
) -> str:
    if weights.empty:
        return "empty_weights"
    if abs(float(weights.sum()) - 1.0) > 1e-4:
        return "weight_sum_not_one"
    if float(weights.min()) < -1e-8:
        return "negative_weight"
    raw_risky_weights = weights.drop(labels=[CASH_CODE], errors="ignore").clip(lower=0.0).astype(float)
    dust_weights = raw_risky_weights[(raw_risky_weights > 1e-8) & (raw_risky_weights < MIN_ACTIVE_WEIGHT - 1e-8)]
    if not dust_weights.empty:
        return "single_stock_below_min_weight"
    risky_weights = effective_risky_weights(weights)
    if len(risky_weights) > hard_constraints.diversification_target:
        return "holding_count_exceeds_target"
    if not risky_weights.empty and float(risky_weights.max()) > hard_constraints.max_single_weight + 1e-4:
        return "single_stock_exceeds_max_weight"
    if prev_weights is not None:
        prev_cash = max(float(prev_weights.get(CASH_CODE, 0.0)), 0.0)
        min_cash = max(0.0, prev_cash * (1.0 - hard_constraints.risk_budget))
        cash_weight = max(float(weights.get(CASH_CODE, 0.0)), 0.0)
        if cash_weight + 1e-4 < min_cash:
            return "cash_deployment_exceeds_risk_budget"
        risky_prev = effective_risky_weights(prev_weights)
        all_codes = sorted(set(risky_prev.index).union(risky_weights.index))
        risky_prev = risky_prev.reindex(all_codes, fill_value=0.0)
        risky_new = risky_weights.reindex(all_codes, fill_value=0.0)
        turnover = float((risky_new - risky_prev).abs().sum())
        if turnover > hard_constraints.turnover_cap + 1e-4:
            return "turnover_exceeds_cap"
    return ""


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


def _frame_scalar(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.iloc[0])


def _build_contextual_feature_matrix(
    frame: pd.DataFrame,
    preference: PreferenceVector,
    news_context: dict[str, float],
) -> np.ndarray:
    sentiment = news_context["news_sentiment_score"]
    pressure = news_context["news_risk_pressure"]
    attention = min(1.0, news_context["news_attention"] / math.log1p(20.0))
    positive_mood = max(sentiment, 0.0) * attention
    negative_mood = max(-sentiment, 0.0) * attention
    style_bias = {
        "momentum": np.array([0.30, 0.00, 0.00, -0.05, 0.05, 0.00, 0.00, 0.15, 0.15, -0.05], dtype=float),
        "value": np.array([0.00, 0.30, 0.05, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, -0.05], dtype=float),
        "quality": np.array([0.00, 0.00, 0.30, 0.10, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00], dtype=float),
        "low_vol": np.array([-0.05, 0.00, 0.10, 0.35, 0.00, 0.00, 0.00, -0.05, 0.00, 0.10], dtype=float),
        "balanced": np.array([0.00, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20], dtype=float),
    }[preference.style_tilt]
    news_bias = np.array(
        [
            0.22 * positive_mood - 0.18 * pressure * attention,
            0.00,
            0.18 * negative_mood,
            0.24 * negative_mood,
            0.00,
            0.00,
            0.00,
            0.12 * positive_mood,
            0.12 * positive_mood,
            0.14 * pressure,
        ],
        dtype=float,
    )
    column_scale = 1.0 + style_bias + news_bias
    return frame[POLICY_FEATURE_COLUMNS].to_numpy(dtype=float) * column_scale


def _cap_risky_weights_preserve_sum(
    risky_weights: pd.Series,
    max_weight: float,
) -> pd.Series:
    risky_weights = risky_weights.clip(lower=0.0).astype(float)
    max_weight = float(max(1e-6, min(1.0, max_weight)))
    for _ in range(8):
        over = risky_weights > max_weight
        if not over.any():
            return risky_weights
        excess = float((risky_weights[over] - max_weight).sum())
        risky_weights.loc[over] = max_weight
        under = risky_weights < max_weight - 1e-12
        if not under.any() or excess <= 1e-12:
            return risky_weights
        room = max_weight - risky_weights[under]
        room_sum = float(room.sum())
        if room_sum <= 1e-12:
            return risky_weights
        risky_weights.loc[under] += excess * (room / room_sum)
    return risky_weights
