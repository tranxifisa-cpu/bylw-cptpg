from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import digamma

from .config import ExperimentConfig
from .metrics import DYNAMIC_LOCAL_REGRET_ALPHA, dynamic_local_regret_value, smoothed_gradient
from .market_data import TushareDataClient
from .strategies import (
    compute_expected_return_gradient,
    compute_exponential_utility_gradient,
    compute_exponential_utility_objective,
    compute_cpt_gradient,
    compute_cpt_objective,
    dirichlet_alpha_from_scores,
    normalize_latent_vector,
    normalized_update_direction,
    transaction_remainder_factor,
    update_reference_point,
)


SyntheticPolicy = Literal["softmax", "sparsemax", "dirichlet"]
DirichletExecutionMode = Literal["sample", "mean"]
SyntheticMethod = Literal[
    "dynamic_cpt_pg",
    "symmetric_cpt_pg",
    "static_cpt_pg",
    "expected_return_pg",
    "exponential_utility_pg",
]


SYNTHETIC_FEATURE_COLUMNS = [
    "bias",
    "cash_indicator",
    "momentum_signal",
    "reversal_signal",
    "low_vol_signal",
    "value_signal",
    "turnover_signal",
    "quality_signal",
    "cycle_signal",
    "prev_weight_signal",
]


@dataclass(frozen=True)
class SyntheticMarketConfig:
    steps: int = 500
    prewarm_steps: int = 30
    risky_assets: int = 2
    seed: int = 3407
    start_date: str = "20230101"
    end_date: str = "20260531"
    index_code: str = "000300.SH"
    return_bound: float = 0.08
    transaction_cost_bps: float = 0.0
    factor_strength: float = 2.5


@dataclass
class SyntheticMarket:
    config: SyntheticMarketConfig
    returns: np.ndarray
    price_relatives: np.ndarray
    features: np.ndarray
    drift: np.ndarray
    environment_drift: np.ndarray
    asset_codes: list[str]
    regime_labels: list[str]

    @property
    def total_steps(self) -> int:
        return int(self.returns.shape[0])

    @staticmethod
    def build(config: SyntheticMarketConfig) -> "SyntheticMarket":
        return SyntheticMarket._build_semi_hs300(config)

    @staticmethod
    def _build_semi_hs300(config: SyntheticMarketConfig) -> "SyntheticMarket":
        total_steps = config.prewarm_steps + config.steps
        if total_steps <= 30:
            raise RuntimeError("Semi-synthetic HS300 environment requires more than 30 total steps")
        client = TushareDataClient(ExperimentConfig())
        trade_dates = client.trade_calendar(config.start_date, config.end_date)
        if len(trade_dates) < total_steps + 30:
            raise RuntimeError(
                f"Not enough Tushare trade dates for semi-synthetic HS300 environment: "
                f"need at least {total_steps + 30}, got {len(trade_dates)}"
            )
        weights = client.index_weight(config.index_code, config.start_date, config.end_date)
        if weights.empty or "con_code" not in weights.columns:
            raise RuntimeError(f"Tushare index_weight returned no constituents for {config.index_code}")
        latest_weight_date = str(weights["trade_date"].astype(str).max())
        latest_weights = weights[weights["trade_date"].astype(str) == latest_weight_date].copy()
        latest_weights["weight"] = pd.to_numeric(latest_weights["weight"], errors="coerce").fillna(0.0)
        hs300_codes = latest_weights.sort_values("weight", ascending=False)["con_code"].astype(str).drop_duplicates().tolist()
        if not hs300_codes:
            raise RuntimeError(f"No HS300 constituent codes found at index weight date {latest_weight_date}")

        # Keep enough earlier dates to compute strictly lagged factors before the evaluation window.
        candidate_dates = trade_dates[-(total_steps + 60):]
        daily_frames: list[pd.DataFrame] = []
        basic_frames: list[pd.DataFrame] = []
        hs300_set = set(hs300_codes)
        for trade_date in candidate_dates:
            daily = client.daily_by_trade_date(trade_date)
            basic = client.daily_basic_by_trade_date(trade_date)
            if daily.empty or basic.empty:
                continue
            daily = daily[daily["ts_code"].astype(str).isin(hs300_set)].copy()
            basic = basic[basic["ts_code"].astype(str).isin(hs300_set)].copy()
            if daily.empty or basic.empty:
                continue
            daily_frames.append(daily)
            basic_frames.append(basic)
        if not daily_frames or not basic_frames:
            raise RuntimeError("No cached Tushare daily/daily_basic frames are available for HS300 semi-synthetic build")

        daily_panel = pd.concat(daily_frames, ignore_index=True)
        basic_panel = pd.concat(basic_frames, ignore_index=True)
        return_matrix = (
            daily_panel.pivot_table(index="trade_date", columns="ts_code", values="pct_chg", aggfunc="last")
            .sort_index()
            / 100.0
        )
        amount_matrix = daily_panel.pivot_table(index="trade_date", columns="ts_code", values="amount", aggfunc="last").sort_index()
        pb_matrix = basic_panel.pivot_table(index="trade_date", columns="ts_code", values="pb", aggfunc="last").sort_index()
        turnover_matrix = basic_panel.pivot_table(index="trade_date", columns="ts_code", values="turnover_rate", aggfunc="last").sort_index()
        volume_ratio_matrix = basic_panel.pivot_table(index="trade_date", columns="ts_code", values="volume_ratio", aggfunc="last").sort_index()
        circ_mv_matrix = basic_panel.pivot_table(index="trade_date", columns="ts_code", values="circ_mv", aggfunc="last").sort_index()

        common_dates = sorted(
            set(return_matrix.index)
            .intersection(amount_matrix.index)
            .intersection(pb_matrix.index)
            .intersection(turnover_matrix.index)
            .intersection(volume_ratio_matrix.index)
            .intersection(circ_mv_matrix.index)
        )
        return_matrix = return_matrix.loc[common_dates]
        amount_matrix = amount_matrix.loc[common_dates]
        pb_matrix = pb_matrix.loc[common_dates]
        turnover_matrix = turnover_matrix.loc[common_dates]
        volume_ratio_matrix = volume_ratio_matrix.loc[common_dates]
        circ_mv_matrix = circ_mv_matrix.loc[common_dates]

        eligible_codes = [
            code
            for code in hs300_codes
            if code in return_matrix.columns
            and return_matrix[code].notna().all()
            and amount_matrix[code].notna().all()
            and pb_matrix[code].notna().all()
            and turnover_matrix[code].notna().all()
            and volume_ratio_matrix[code].notna().all()
            and circ_mv_matrix[code].notna().all()
        ]
        if len(eligible_codes) < 2:
            raise RuntimeError("Fewer than two HS300 constituents have complete daily and daily_basic data")
        selected_codes = eligible_codes[: min(int(config.risky_assets), len(eligible_codes))]
        if len(common_dates) < total_steps + 30:
            raise RuntimeError(f"Not enough complete HS300 dates after cleaning: need {total_steps + 30}, got {len(common_dates)}")

        selected_dates = common_dates[-(total_steps + 30):]
        real_returns_full = return_matrix.loc[selected_dates, selected_codes].astype(float)
        pb_full = pb_matrix.loc[selected_dates, selected_codes].astype(float)
        turnover_full = turnover_matrix.loc[selected_dates, selected_codes].astype(float)
        volume_ratio_full = volume_ratio_matrix.loc[selected_dates, selected_codes].astype(float)
        circ_mv_full = circ_mv_matrix.loc[selected_dates, selected_codes].astype(float)

        momentum_20 = real_returns_full.rolling(20).mean().shift(1)
        reversal_5 = -real_returns_full.rolling(5).mean().shift(1)
        low_vol_20 = -real_returns_full.rolling(20).std(ddof=0).shift(1)
        value_pb = -np.log(pb_full.where(pb_full > 0.0)).shift(1)
        turnover_signal = np.log1p(turnover_full).shift(1)
        quality_signal = np.log(circ_mv_full.where(circ_mv_full > 0.0)).shift(1) - np.abs(
            volume_ratio_full.shift(1).fillna(1.0) - 1.0
        )
        market_cycle = real_returns_full.mean(axis=1).rolling(20).mean().shift(1)

        factor_frames = {
            "momentum_signal": momentum_20,
            "reversal_signal": reversal_5,
            "low_vol_signal": low_vol_20,
            "value_signal": value_pb,
            "turnover_signal": turnover_signal,
            "quality_signal": quality_signal,
        }
        valid_mask = pd.Series(True, index=real_returns_full.index)
        for frame in factor_frames.values():
            valid_mask &= frame.notna().all(axis=1)
        valid_mask &= market_cycle.notna()
        valid_dates = valid_mask[valid_mask].index.tolist()
        if len(valid_dates) < total_steps:
            raise RuntimeError(f"Not enough lagged-factor dates after rolling windows: need {total_steps}, got {len(valid_dates)}")
        final_dates = valid_dates[-total_steps:]
        real_returns = real_returns_full.loc[final_dates].to_numpy(dtype=float)
        feature_values = {name: _cross_sectional_zscore(frame.loc[final_dates]) for name, frame in factor_frames.items()}
        cycle_values = np.tanh((market_cycle.loc[final_dates].to_numpy(dtype=float) / 0.02).reshape(-1, 1))

        risky_assets = len(selected_codes)
        feature_index = {name: idx for idx, name in enumerate(SYNTHETIC_FEATURE_COLUMNS)}
        features = np.zeros((total_steps, risky_assets + 1, len(SYNTHETIC_FEATURE_COLUMNS)), dtype=float)
        features[:, :, feature_index["bias"]] = 1.0
        features[:, 0, feature_index["cash_indicator"]] = 1.0
        for name, values in feature_values.items():
            features[:, 1:, feature_index[name]] = values
        features[:, :, feature_index["cycle_signal"]] = cycle_values
        features[:, :, feature_index["prev_weight_signal"]] = 0.0

        regimes = _regime_labels(total_steps, config.prewarm_steps)
        expected = np.zeros((total_steps, risky_assets), dtype=float)
        noise_scale = np.zeros(total_steps, dtype=float)
        for idx, regime in enumerate(regimes):
            coefs, market_mu, scale = _semi_synthetic_regime_parameters(regime)
            factor_component = np.zeros(risky_assets, dtype=float)
            for name, coef in coefs.items():
                factor_component += coef * features[idx, 1:, feature_index[name]]
            expected[idx] = market_mu + float(config.factor_strength) * factor_component
            noise_scale[idx] = scale

        rng = np.random.default_rng(config.seed)
        market_real = real_returns.mean(axis=1, keepdims=True)
        centered_real = real_returns - market_real
        common_noise = market_real - float(np.mean(market_real))
        idiosyncratic = centered_real * noise_scale.reshape(-1, 1)
        small_noise = rng.normal(0.0, 0.0015, size=expected.shape)
        risky_returns = expected + idiosyncratic + common_noise * noise_scale.reshape(-1, 1) + small_noise
        risky_returns = config.return_bound * np.tanh(risky_returns / config.return_bound)

        returns = np.zeros((total_steps, risky_assets + 1), dtype=float)
        returns[:, 1:] = risky_returns
        price_relatives = 1.0 + returns
        environment_drift = np.zeros(total_steps, dtype=float)
        environment_drift[1:] = np.linalg.norm(expected[1:] - expected[:-1], axis=1)
        asset_codes = ["CASH", *selected_codes]
        return SyntheticMarket(
            config=config,
            returns=returns,
            price_relatives=price_relatives,
            features=np.clip(features, -1.0, 1.0),
            drift=expected,
            environment_drift=environment_drift,
            asset_codes=asset_codes,
            regime_labels=regimes,
        )


def _cross_sectional_zscore(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    std = np.where(std <= 1e-12, 1.0, std)
    return np.clip((values - mean) / std, -3.0, 3.0) / 3.0


def _regime_labels(total_steps: int, prewarm_steps: int) -> list[str]:
    cycle_steps = 500
    uptrend_end = int(round(0.30 * cycle_steps))
    oscillation_end = int(round(0.58 * cycle_steps))
    shock_end = int(round(0.70 * cycle_steps))
    labels: list[str] = []
    for step in range(total_steps):
        eval_step = step - prewarm_steps + 1
        if eval_step <= 0:
            labels.append("prewarm")
            continue
        cycle_step = ((eval_step - 1) % cycle_steps) + 1
        if cycle_step <= uptrend_end:
            labels.append("uptrend_momentum")
        elif cycle_step <= oscillation_end:
            labels.append("oscillation_reversal")
        elif cycle_step <= shock_end:
            labels.append("negative_shock")
        else:
            labels.append("style_switch_lowvol_quality")
    return labels


def _semi_synthetic_regime_parameters(regime: str) -> tuple[dict[str, float], float, float]:
    if regime == "uptrend_momentum":
        return (
            {
                "momentum_signal": 0.0040,
                "reversal_signal": -0.0008,
                "low_vol_signal": 0.0004,
                "value_signal": 0.0003,
                "turnover_signal": 0.0004,
                "quality_signal": 0.0005,
            },
            0.0009,
            0.55,
        )
    if regime == "oscillation_reversal":
        return (
            {
                "momentum_signal": -0.0006,
                "reversal_signal": 0.0035,
                "low_vol_signal": 0.0006,
                "value_signal": 0.0002,
                "turnover_signal": 0.0002,
                "quality_signal": 0.0003,
            },
            0.0001,
            0.70,
        )
    if regime == "negative_shock":
        return (
            {
                "momentum_signal": -0.0010,
                "reversal_signal": -0.0006,
                "low_vol_signal": -0.0004,
                "value_signal": -0.0002,
                "turnover_signal": -0.0002,
                "quality_signal": -0.0002,
            },
            -0.0035,
            1.70,
        )
    if regime == "style_switch_lowvol_quality":
        return (
            {
                "momentum_signal": 0.0000,
                "reversal_signal": 0.0002,
                "low_vol_signal": 0.0030,
                "value_signal": 0.0010,
                "turnover_signal": 0.0001,
                "quality_signal": 0.0022,
            },
            0.0004,
            0.75,
        )
    return (
        {
            "momentum_signal": 0.0000,
            "reversal_signal": 0.0000,
            "low_vol_signal": 0.0000,
            "value_signal": 0.0000,
            "turnover_signal": 0.0000,
            "quality_signal": 0.0000,
        },
        0.0,
        0.60,
    )


@dataclass
class SyntheticStep:
    step: int
    policy: str
    method: str
    seed: int
    market_regime: str
    portfolio_return: float
    normalized_wealth: float
    relative_wealth: float
    wealth: float
    cash_weight: float
    turnover: float
    pgr: float
    plr: float
    disposition_spread: float
    realized_gain_count: int
    realized_loss_count: int
    paper_gain_count: int
    paper_loss_count: int
    reference_point: float
    reference_drift: float
    objective_estimate: float
    gradient_norm: float
    dynamic_local_regret_term: float
    dynamic_local_regret: float
    average_dynamic_local_regret: float
    squared_gradient_norm: float
    cumulative_squared_gradient_norm: float
    average_squared_gradient_norm: float
    update_norm: float
    theta_norm: float
    theta_max_abs: float
    theta_boundary_share: float
    environment_drift: float
    window_approximation_proxy: float
    cpt_sample_count: int
    gradient_sample_count: int


class SyntheticCPTPGTracker:
    def __init__(
        self,
        *,
        market: SyntheticMarket,
        config: ExperimentConfig,
        policy: SyntheticPolicy,
        seed: int,
        method: SyntheticMethod = "dynamic_cpt_pg",
        static_reference: bool = False,
        dirichlet_execution_mode: DirichletExecutionMode = "sample",
        estimation_mode: Literal["rolling_window", "on_policy_episode"] = "rolling_window",
    ) -> None:
        self.market = market
        self.config = config
        self.policy = policy
        self.method = method
        self.seed = int(seed)
        self.static_reference = bool(static_reference or method == "static_cpt_pg")
        self.dirichlet_execution_mode = dirichlet_execution_mode
        self.estimation_mode = estimation_mode
        if self.config.reference_update_frequency not in ("daily", "episode"):
            raise RuntimeError("reference_update_frequency must be 'daily' or 'episode'")
        self.rng = np.random.default_rng(seed)
        self.theta = np.zeros(len(SYNTHETIC_FEATURE_COLUMNS), dtype=float)
        self.reference_point = 1.0
        self.weights = np.zeros(len(market.asset_codes), dtype=float)
        self.weights[0] = 1.0
        self.open_prices = np.ones(len(market.asset_codes), dtype=float)
        self.cost_basis = np.full(len(market.asset_codes), np.nan, dtype=float)
        self.cost_basis[0] = 1.0
        self.wealth = float(config.initial_capital_amount)
        self.initial_wealth = float(config.initial_capital_amount)
        self.cumulative_squared_gradient_norm = 0.0
        self.cumulative_dynamic_local_regret = 0.0
        self.gradient_history: list[np.ndarray] = []

    def run(self) -> pd.DataFrame:
        if self.estimation_mode == "on_policy_episode":
            return self._run_on_policy_episodes()
        if self.estimation_mode != "rolling_window":
            raise RuntimeError(f"Unknown synthetic estimation mode: {self.estimation_mode}")
            return self._run_rolling_window()

    def _gradient_smoothing_window(self) -> int:
        configured = self.config.gradient_smoothing_window
        if configured is None:
            configured = self.config.evaluation_horizon
        return max(1, int(configured))

    def _run_rolling_window(self) -> pd.DataFrame:
        rows: list[SyntheticStep] = []
        start = self.market.config.prewarm_steps
        end = start + self.market.config.steps
        for eval_index, step in enumerate(range(start, end), start=1):
            gradient, objective = self._estimate_gradient_and_objective(step)
            gamma_t = float(self.config.gamma0)
            smoothed_update_gradient = smoothed_gradient(
                self.gradient_history,
                gradient,
                window=self._gradient_smoothing_window(),
                alpha=DYNAMIC_LOCAL_REGRET_ALPHA,
            )
            update_direction = (
                normalized_update_direction(smoothed_update_gradient)
                if self.config.normalize_gradient_update
                else np.asarray(smoothed_update_gradient, dtype=float)
            )
            update = gamma_t * update_direction
            theta_before = self.theta.copy()
            self.theta = theta_before + update
            gradient_norm = float(np.linalg.norm(gradient))
            dynamic_local_regret = dynamic_local_regret_value(
                self.gradient_history,
                gradient,
                window=self._gradient_smoothing_window(),
                alpha=DYNAMIC_LOCAL_REGRET_ALPHA,
            )
            self.gradient_history.append(gradient.copy())
            self.cumulative_squared_gradient_norm += gradient_norm * gradient_norm
            self.cumulative_dynamic_local_regret += dynamic_local_regret

            action_weights, _ = self._sample_action(step, self.rng, collect_score=False)
            disposition = self._disposition_effect_metrics(action_weights)
            self._update_cost_basis_after_trade(action_weights)
            portfolio_return, next_weights, turnover = self._apply_return(action_weights, self.market.price_relatives[step])
            next_wealth = self.wealth * (1.0 + portfolio_return)
            normalized_wealth = next_wealth / max(self.initial_wealth, 1e-12)
            previous_reference = self.reference_point
            if self.method in ("dynamic_cpt_pg", "symmetric_cpt_pg"):
                eta_gain = self.config.eta_gain
                eta_loss = self.config.eta_loss
                if self.method == "symmetric_cpt_pg":
                    eta_gain = eta_loss = 0.5 * (eta_gain + eta_loss)
                self.reference_point = update_reference_point(
                    self.reference_point,
                    normalized_wealth,
                    eta_gain,
                    eta_loss,
                )
            self.weights = next_weights
            self.wealth = next_wealth

            rows.append(
                SyntheticStep(
                    step=eval_index,
                    policy=self.policy,
                    method=self.method,
                    seed=self.seed,
                    market_regime=self.market.regime_labels[step],
                    portfolio_return=float(portfolio_return),
                    normalized_wealth=float(normalized_wealth),
                    relative_wealth=float(normalized_wealth - previous_reference),
                    wealth=float(self.wealth),
                    cash_weight=float(action_weights[0]),
                    turnover=float(turnover),
                    pgr=float(disposition["pgr"]),
                    plr=float(disposition["plr"]),
                    disposition_spread=float(disposition["disposition_spread"]),
                    realized_gain_count=int(disposition["realized_gain_count"]),
                    realized_loss_count=int(disposition["realized_loss_count"]),
                    paper_gain_count=int(disposition["paper_gain_count"]),
                    paper_loss_count=int(disposition["paper_loss_count"]),
                    reference_point=float(self.reference_point),
                    reference_drift=float(abs(self.reference_point - previous_reference)),
                    objective_estimate=float(objective),
                    gradient_norm=gradient_norm,
                    dynamic_local_regret_term=float(dynamic_local_regret),
                    dynamic_local_regret=float(self.cumulative_dynamic_local_regret),
                    average_dynamic_local_regret=float(self.cumulative_dynamic_local_regret / eval_index),
                    squared_gradient_norm=float(gradient_norm * gradient_norm),
                    cumulative_squared_gradient_norm=float(self.cumulative_squared_gradient_norm),
                    average_squared_gradient_norm=float(self.cumulative_squared_gradient_norm / eval_index),
                    update_norm=float(np.linalg.norm(self.theta - theta_before)),
                    theta_norm=float(np.linalg.norm(self.theta)),
                    theta_max_abs=float(np.max(np.abs(self.theta))),
                    theta_boundary_share=0.0,
                    environment_drift=float(self.market.environment_drift[step]),
                    window_approximation_proxy=float(self._window_approximation_proxy(step)),
                    cpt_sample_count=int(
                        self.config.gradient_sample_base
                        if self.config.shared_cpt_gradient_samples
                        else self.config.cpt_sample_base
                    ),
                    gradient_sample_count=int(self.config.gradient_sample_base),
                )
            )
        return pd.DataFrame([row.__dict__ for row in rows])

    def _run_on_policy_episodes(self) -> pd.DataFrame:
        rows: list[SyntheticStep] = []
        start = self.market.config.prewarm_steps
        end = start + self.market.config.steps
        horizon = int(self.config.evaluation_horizon)
        if horizon <= 0:
            raise RuntimeError("evaluation_horizon must be positive for on_policy_episode mode")
        episode_index = 0
        for block_start in range(start, end, horizon):
            block_steps = list(range(block_start, min(block_start + horizon, end)))
            if not block_steps:
                continue
            episode_index += 1
            theta_before = self.theta.copy()
            start_weights = self.weights.copy()
            start_normalized_wealth = self.wealth / max(self.initial_wealth, 1e-12)
            episode_reference = self.reference_point
            start_wealth = self.wealth
            cash_values: list[float] = []
            turnovers: list[float] = []
            realized_gain_counts: list[int] = []
            realized_loss_counts: list[int] = []
            paper_gain_counts: list[int] = []
            paper_loss_counts: list[int] = []
            weighted_regime_drift: list[float] = []

            for market_step in block_steps:
                action_weights, _ = self._sample_action(market_step, self.rng, collect_score=False)
                disposition = self._disposition_effect_metrics(action_weights)
                self._update_cost_basis_after_trade(action_weights)
                portfolio_return, next_weights, turnover = self._apply_return(
                    action_weights,
                    self.market.price_relatives[market_step],
                )
                self.wealth *= 1.0 + portfolio_return
                self.weights = next_weights
                normalized_step_wealth = self.wealth / max(self.initial_wealth, 1e-12)
                if (
                    self.config.reference_update_frequency == "daily"
                    and self.method in ("dynamic_cpt_pg", "symmetric_cpt_pg")
                ):
                    eta_gain = self.config.eta_gain
                    eta_loss = self.config.eta_loss
                    if self.method == "symmetric_cpt_pg":
                        eta_gain = eta_loss = 0.5 * (eta_gain + eta_loss)
                    self.reference_point = update_reference_point(
                        self.reference_point,
                        normalized_step_wealth,
                        eta_gain,
                        eta_loss,
                    )
                cash_values.append(float(action_weights[0]))
                turnovers.append(float(turnover))
                realized_gain_counts.append(int(disposition["realized_gain_count"]))
                realized_loss_counts.append(int(disposition["realized_loss_count"]))
                paper_gain_counts.append(int(disposition["paper_gain_count"]))
                paper_loss_counts.append(int(disposition["paper_loss_count"]))
                weighted_regime_drift.append(float(self.market.environment_drift[market_step]))

            if (
                self.config.reference_update_frequency == "episode"
                and self.method in ("dynamic_cpt_pg", "symmetric_cpt_pg")
            ):
                eta_gain = self.config.eta_gain
                eta_loss = self.config.eta_loss
                if self.method == "symmetric_cpt_pg":
                    eta_gain = eta_loss = 0.5 * (eta_gain + eta_loss)
                self.reference_point = update_reference_point(
                    self.reference_point,
                    self.wealth / max(self.initial_wealth, 1e-12),
                    eta_gain,
                    eta_loss,
                )

            gradient, objective = self._estimate_gradient_and_objective_from_history(
                history=block_steps,
                reference_point=episode_reference,
                start_normalized_wealth=start_normalized_wealth,
                start_weights=start_weights,
                seed_step=block_start,
            )
            gamma_t = float(self.config.gamma0)
            smoothed_update_gradient = smoothed_gradient(
                self.gradient_history,
                gradient,
                window=self._gradient_smoothing_window(),
                alpha=DYNAMIC_LOCAL_REGRET_ALPHA,
            )
            update_direction = (
                normalized_update_direction(smoothed_update_gradient)
                if self.config.normalize_gradient_update
                else np.asarray(smoothed_update_gradient, dtype=float)
            )
            update = gamma_t * update_direction
            self.theta = theta_before + update
            gradient_norm = float(np.linalg.norm(gradient))
            dynamic_local_regret = dynamic_local_regret_value(
                self.gradient_history,
                gradient,
                window=self._gradient_smoothing_window(),
                alpha=DYNAMIC_LOCAL_REGRET_ALPHA,
            )
            self.gradient_history.append(gradient.copy())
            self.cumulative_squared_gradient_norm += gradient_norm * gradient_norm
            self.cumulative_dynamic_local_regret += dynamic_local_regret

            previous_reference = episode_reference
            normalized_wealth = self.wealth / max(self.initial_wealth, 1e-12)
            episode_realized_gain_count = int(np.sum(realized_gain_counts))
            episode_realized_loss_count = int(np.sum(realized_loss_counts))
            episode_paper_gain_count = int(np.sum(paper_gain_counts))
            episode_paper_loss_count = int(np.sum(paper_loss_counts))
            episode_pgr_denominator = episode_realized_gain_count + episode_paper_gain_count
            episode_plr_denominator = episode_realized_loss_count + episode_paper_loss_count
            episode_pgr = (
                0.0
                if episode_pgr_denominator == 0
                else float(episode_realized_gain_count / episode_pgr_denominator)
            )
            episode_plr = (
                0.0
                if episode_plr_denominator == 0
                else float(episode_realized_loss_count / episode_plr_denominator)
            )

            rows.append(
                SyntheticStep(
                    step=episode_index,
                    policy=self.policy,
                    method=self.method,
                    seed=self.seed,
                    market_regime=self.market.regime_labels[block_steps[0]],
                    portfolio_return=float(self.wealth / max(start_wealth, 1e-12) - 1.0),
                    normalized_wealth=float(normalized_wealth),
                    relative_wealth=float(normalized_wealth - previous_reference),
                    wealth=float(self.wealth),
                    cash_weight=float(np.mean(cash_values)),
                    turnover=float(np.mean(turnovers)),
                    pgr=episode_pgr,
                    plr=episode_plr,
                    disposition_spread=float(episode_pgr - episode_plr),
                    realized_gain_count=episode_realized_gain_count,
                    realized_loss_count=episode_realized_loss_count,
                    paper_gain_count=episode_paper_gain_count,
                    paper_loss_count=episode_paper_loss_count,
                    reference_point=float(self.reference_point),
                    reference_drift=float(abs(self.reference_point - previous_reference)),
                    objective_estimate=float(objective),
                    gradient_norm=gradient_norm,
                    dynamic_local_regret_term=float(dynamic_local_regret),
                    dynamic_local_regret=float(self.cumulative_dynamic_local_regret),
                    average_dynamic_local_regret=float(self.cumulative_dynamic_local_regret / episode_index),
                    squared_gradient_norm=float(gradient_norm * gradient_norm),
                    cumulative_squared_gradient_norm=float(self.cumulative_squared_gradient_norm),
                    average_squared_gradient_norm=float(self.cumulative_squared_gradient_norm / episode_index),
                    update_norm=float(np.linalg.norm(self.theta - theta_before)),
                    theta_norm=float(np.linalg.norm(self.theta)),
                    theta_max_abs=float(np.max(np.abs(self.theta))),
                    theta_boundary_share=0.0,
                    environment_drift=float(np.mean(weighted_regime_drift)),
                    window_approximation_proxy=0.0,
                    cpt_sample_count=int(
                        self.config.gradient_sample_base
                        if self.config.shared_cpt_gradient_samples
                        else self.config.cpt_sample_base
                    ),
                    gradient_sample_count=int(self.config.gradient_sample_base),
                )
            )
        return pd.DataFrame([row.__dict__ for row in rows])

    def set_state(
        self,
        *,
        normalized_wealth: float,
        reference_point: float,
        theta: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        self.wealth = float(normalized_wealth) * self.initial_wealth
        self.reference_point = float(reference_point)
        if theta is not None:
            theta_array = np.asarray(theta, dtype=float)
            if theta_array.shape != self.theta.shape:
                raise RuntimeError("Synthetic tracker theta state has incompatible shape")
            self.theta = theta_array.copy()
        if weights is not None:
            weight_array = np.asarray(weights, dtype=float)
            if weight_array.shape != self.weights.shape:
                raise RuntimeError("Synthetic tracker weight state has incompatible shape")
            total = float(weight_array.sum())
            if total <= 0.0:
                raise RuntimeError("Synthetic tracker weight state must have positive sum")
            self.weights = np.maximum(weight_array, 0.0) / total
            self.cost_basis = np.where(self.weights > 1e-12, self.open_prices, np.nan)
            self.cost_basis[0] = 1.0

    def _estimate_gradient_and_objective(self, step: int) -> tuple[np.ndarray, float]:
        history = list(range(step - self.config.evaluation_horizon, step))
        return self._estimate_gradient_and_objective_from_history(
            history=history,
            reference_point=self.reference_point,
            start_normalized_wealth=self.wealth / max(self.initial_wealth, 1e-12),
            start_weights=self.weights,
            seed_step=step,
        )

    def _estimate_gradient_and_objective_from_history(
        self,
        *,
        history: list[int],
        reference_point: float,
        start_normalized_wealth: float,
        start_weights: np.ndarray,
        seed_step: int,
    ) -> tuple[np.ndarray, float]:
        track_reference_path = self.method in ("dynamic_cpt_pg", "symmetric_cpt_pg")
        gradient_samples, score_vectors, gradient_references = self._sample_window_outcomes(
            history=history,
            sample_count=int(self.config.gradient_sample_base),
            rng=np.random.default_rng(self._stable_seed(seed_step, "gradient")),
            collect_scores=True,
            start_normalized_wealth=start_normalized_wealth,
            start_weights=start_weights,
            reference_start=reference_point,
            track_reference_path=track_reference_path,
        )
        if self.config.shared_cpt_gradient_samples:
            objective_samples = gradient_samples
            objective_references = gradient_references
        else:
            objective_samples, _, objective_references = self._sample_window_outcomes(
                history=history,
                sample_count=int(self.config.cpt_sample_base),
                rng=np.random.default_rng(self._stable_seed(seed_step, "cpt")),
                collect_scores=False,
                start_normalized_wealth=start_normalized_wealth,
                start_weights=start_weights,
                reference_start=reference_point,
                track_reference_path=track_reference_path,
            )
        if self.method == "expected_return_pg":
            return (
                compute_expected_return_gradient(gradient_samples, score_vectors),
                float(np.mean(objective_samples)),
            )
        if self.method == "exponential_utility_pg":
            return (
                compute_exponential_utility_gradient(
                    gradient_samples,
                    score_vectors,
                    self.config.exponential_risk_aversion,
                ),
                compute_exponential_utility_objective(
                    objective_samples,
                    self.config.exponential_risk_aversion,
                ),
            )
        cpt_relative = objective_samples - (objective_references if objective_references is not None else reference_point)
        gradient_relative = gradient_samples - (gradient_references if gradient_references is not None else reference_point)
        objective = compute_cpt_objective(cpt_relative.tolist(), self.config)
        gradient = compute_cpt_gradient(
            gradient_relative.tolist(),
            [row for row in score_vectors],
            cpt_relative.tolist(),
            self.config,
        )
        return gradient, objective

    def _sample_window_outcomes(
        self,
        *,
        history: list[int],
        sample_count: int,
        rng: np.random.Generator,
        collect_scores: bool,
        start_normalized_wealth: float | None = None,
        start_weights: np.ndarray | None = None,
        reference_start: float | None = None,
        track_reference_path: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        returns = np.zeros(sample_count, dtype=float)
        normalized_start = (
            float(start_normalized_wealth)
            if start_normalized_wealth is not None
            else self.wealth / max(self.initial_wealth, 1e-12)
        )
        growth = np.full(sample_count, normalized_start, dtype=float)
        reference_values = (
            np.full(sample_count, self.reference_point if reference_start is None else float(reference_start), dtype=float)
            if track_reference_path
            else None
        )
        base_weights = np.asarray(start_weights, dtype=float) if start_weights is not None else self.weights
        previous = np.tile(base_weights, (sample_count, 1))
        scores = np.zeros((sample_count, len(SYNTHETIC_FEATURE_COLUMNS)), dtype=float)
        for step in history:
            weight_matrix, score_matrix = self._sample_action_batch(
                step,
                sample_count,
                rng,
                collect_scores,
                prev_weight_matrix=previous,
            )
            mu = transaction_remainder_factor(
                previous,
                weight_matrix,
                self.config.trade_cost_bps / 10000.0,
                cash_index=0,
            )
            step_price_relatives = weight_matrix @ self.market.price_relatives[step]
            growth *= mu * step_price_relatives
            if reference_values is not None and self.config.reference_update_frequency == "daily":
                gain_mask = growth >= reference_values
                eta_gain = self.config.eta_gain
                eta_loss = self.config.eta_loss
                if self.method == "symmetric_cpt_pg":
                    eta_gain = eta_loss = 0.5 * (eta_gain + eta_loss)
                reference_values = np.where(
                    gain_mask,
                    reference_values + eta_gain * (growth - reference_values),
                    reference_values - eta_loss * (reference_values - growth),
                )
            if collect_scores:
                scores += score_matrix
            post = weight_matrix * self.market.price_relatives[step].reshape(1, -1)
            post_sum = np.maximum(post.sum(axis=1), 1e-12)
            previous = post / post_sum.reshape(-1, 1)
        returns = growth
        return returns, scores, reference_values

    def _sample_action_batch(
        self,
        step: int,
        sample_count: int,
        rng: np.random.Generator,
        collect_scores: bool,
        prev_weight_matrix: np.ndarray | None = None,
        execution: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = self._features_for_step(step, prev_weight_matrix)
        scores = features @ self.theta
        if self.policy == "dirichlet":
            local_config = self._dirichlet_config()
            alpha, alpha_derivative = dirichlet_alpha_from_scores(scores, local_config)
            if execution and self.dirichlet_execution_mode == "mean":
                weights = np.tile(alpha / np.sum(alpha), (sample_count, 1))
            else:
                weights = rng.dirichlet(alpha, size=sample_count)
            score_matrix = (
                self._dirichlet_score_gradient_matrix(features, weights, alpha, alpha_derivative)
                if collect_scores
                else np.zeros((sample_count, len(SYNTHETIC_FEATURE_COLUMNS)), dtype=float)
            )
            return weights, score_matrix

        if self.config.policy_noise_scale <= 0:
            raise RuntimeError("policy_noise_scale must be positive for softmax and sparsemax synthetic policies")
        noise = rng.normal(0.0, self.config.policy_noise_scale, size=(sample_count, len(scores)))
        latent = scores.reshape(1, -1) + noise
        weights = np.vstack(
            [normalize_latent_vector(row, self.policy, self.config.policy_temperature) for row in latent]
        )
        latent_score = (latent - scores.reshape(1, -1)) / (self.config.policy_noise_scale**2)
        score_matrix = latent_score @ features if collect_scores else np.zeros((sample_count, len(SYNTHETIC_FEATURE_COLUMNS)), dtype=float)
        return weights, score_matrix

    def _sample_action(
        self,
        step: int,
        rng: np.random.Generator,
        *,
        collect_score: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        weights, scores = self._sample_action_batch(
            step,
            1,
            rng,
            collect_score,
            prev_weight_matrix=self.weights.reshape(1, -1),
            execution=True,
        )
        return weights[0], scores[0]

    def _disposition_effect_metrics(self, target_weights: np.ndarray) -> dict[str, float]:
        return disposition_effect_metrics(
            previous_weights=self.weights,
            target_weights=target_weights,
            open_prices=self.open_prices,
            cost_basis=self.cost_basis,
        )

    def _update_cost_basis_after_trade(self, target_weights: np.ndarray) -> None:
        previous = np.asarray(self.weights, dtype=float)
        target = np.asarray(target_weights, dtype=float)
        if previous.shape != target.shape or previous.shape != self.open_prices.shape:
            raise RuntimeError("Cost-basis update inputs must have the same shape")
        prices = np.maximum(self.open_prices, 1e-12)
        updated = self.cost_basis.copy()
        for index in range(1, len(target)):
            if target[index] <= 1e-12:
                updated[index] = np.nan
                continue
            if previous[index] <= 1e-12 or not np.isfinite(updated[index]):
                updated[index] = prices[index]
                continue
            if target[index] > previous[index] + 1e-12:
                previous_shares = previous[index] / prices[index]
                bought_shares = (target[index] - previous[index]) / prices[index]
                total_shares = previous_shares + bought_shares
                if total_shares <= 1e-12:
                    updated[index] = prices[index]
                else:
                    updated[index] = (
                        previous_shares * updated[index] + bought_shares * prices[index]
                    ) / total_shares
        updated[0] = 1.0
        self.cost_basis = updated

    def _apply_return(self, target_weights: np.ndarray, price_relatives: np.ndarray) -> tuple[float, np.ndarray, float]:
        previous = self.weights.reshape(1, -1)
        target = target_weights.reshape(1, -1)
        mu = float(
            transaction_remainder_factor(
                previous,
                target,
                self.config.trade_cost_bps / 10000.0,
                cash_index=0,
            )
        )
        gross_relative = float(target_weights @ price_relatives)
        net_return = mu * gross_relative - 1.0
        post = target_weights * price_relatives
        post_sum = max(float(post.sum()), 1e-12)
        turnover = float(0.5 * np.abs(target_weights - self.weights).sum())
        self.open_prices = np.maximum(self.open_prices * np.asarray(price_relatives, dtype=float), 1e-12)
        self.open_prices[0] = 1.0
        return net_return, post / post_sum, turnover

    def _features_for_step(self, step: int, prev_weight_matrix: np.ndarray | None) -> np.ndarray:
        features = self.market.features[step].copy()
        if prev_weight_matrix is None:
            previous = self.weights
        else:
            previous = np.asarray(prev_weight_matrix, dtype=float).mean(axis=0)
        features[:, SYNTHETIC_FEATURE_COLUMNS.index("prev_weight_signal")] = np.clip(previous, 0.0, 1.0)
        return features

    def _window_approximation_proxy(self, step: int) -> float:
        start = step - self.config.evaluation_horizon
        window_mean = self.market.drift[start:step].mean(axis=0)
        current = self.market.drift[step]
        return float(np.linalg.norm(current - window_mean))

    def _stable_seed(self, step: int, namespace: str) -> int:
        offset = 9176 if namespace == "cpt" else 3191
        return int((self.seed * 1_000_003 + step * 10_007 + offset) % (2**32 - 1))

    def _dirichlet_config(self) -> ExperimentConfig:
        return self.config

    @staticmethod
    def _dirichlet_score_gradient_matrix(
        feature_matrix: np.ndarray,
        weight_matrix: np.ndarray,
        alpha: np.ndarray,
        alpha_derivative: np.ndarray,
    ) -> np.ndarray:
        safe_weights = np.maximum(np.asarray(weight_matrix, dtype=float), 1e-300)
        coefficient_matrix = (
            digamma(float(np.sum(alpha))) - digamma(alpha.reshape(1, -1)) + np.log(safe_weights)
        ) * alpha_derivative.reshape(1, -1)
        return coefficient_matrix @ np.asarray(feature_matrix, dtype=float)


def save_synthetic_plots(trace: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    figures = [
        (
            "tracking_diagnostics.png",
            [
                ("gradient_norm", "Gradient Norm"),
                ("dynamic_local_regret_term", "Dynamic Local Regret Term"),
                ("dynamic_local_regret", "Dynamic Local Regret"),
                ("average_squared_gradient_norm", "Average Squared Gradient"),
                ("cumulative_squared_gradient_norm", "Cumulative Squared Gradient"),
            ],
        ),
        (
            "objective_and_wealth.png",
            [
                ("objective_estimate", "CPT Objective Estimate"),
                ("normalized_wealth", "Normalized Wealth"),
                ("wealth", "Wealth"),
                ("reference_point", "Reference Point"),
                ("relative_wealth", "Relative Wealth"),
            ],
        ),
        (
            "disposition_effect.png",
            [
                ("pgr", "Proportion of Gains Realized"),
                ("plr", "Proportion of Losses Realized"),
                ("disposition_spread", "Disposition Spread"),
            ],
        ),
    ]
    for filename, panels in figures:
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 7), sharex=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (column, title) in zip(axes, panels):
            ax.plot(trace["step"], trace[column], linewidth=1.3)
            ax.set_title(title)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Period")
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def save_synthetic_seed_aggregate_plots(traces: list[pd.DataFrame], output_dir: Path, label: str) -> list[Path]:
    if not traces:
        return []
    merged = pd.concat(traces, ignore_index=True)
    panels = [
        ("gradient_norm", "Gradient Norm"),
        ("dynamic_local_regret_term", "Dynamic Local Regret Term"),
        ("average_dynamic_local_regret", "Average Dynamic Local Regret"),
        ("dynamic_local_regret", "Dynamic Local Regret"),
        ("average_squared_gradient_norm", "Average Squared Gradient"),
        ("cumulative_squared_gradient_norm", "Cumulative Squared Gradient"),
        ("objective_estimate", "CPT Objective Estimate"),
        ("normalized_wealth", "Normalized Wealth"),
        ("wealth", "Wealth"),
        ("reference_point", "Reference Point"),
        ("relative_wealth", "Relative Wealth"),
        ("disposition_spread", "Disposition Spread"),
    ]
    return [_plot_quantile_panels(merged, output_dir / "aggregate_seed_quantiles.png", panels, label)]


def save_synthetic_method_comparison_plots(traces: list[pd.DataFrame], output_dir: Path) -> list[Path]:
    if not traces:
        return []
    merged = pd.concat(traces, ignore_index=True)
    panels = [
        ("average_squared_gradient_norm", "Average Squared Gradient"),
        ("cumulative_squared_gradient_norm", "Cumulative Squared Gradient"),
        ("dynamic_local_regret", "Dynamic Local Regret"),
        ("gradient_norm", "Gradient Norm"),
        ("objective_estimate", "Objective Estimate"),
        ("normalized_wealth", "Normalized Wealth"),
        ("reference_point", "Reference Point"),
        ("disposition_spread", "Disposition Spread"),
    ]
    return [
        _plot_grouped_quantile_panels(
            merged,
            output_dir / "method_comparison_seed_quantiles.png",
            panels,
            "method",
            "semi-synthetic methods",
        )
    ]


def save_synthetic_per_method_plots(traces: list[pd.DataFrame], output_dir: Path) -> list[Path]:
    if not traces:
        return []
    merged = pd.concat(traces, ignore_index=True)
    panels = [
        ("normalized_wealth", "Normalized Wealth"),
        ("objective_estimate", "Objective Estimate"),
        ("gradient_norm", "Gradient Norm"),
        ("dynamic_local_regret_term", "Dynamic Local Regret Term"),
        ("average_dynamic_local_regret", "Average Dynamic Local Regret"),
        ("dynamic_local_regret", "Dynamic Local Regret"),
        ("average_squared_gradient_norm", "Average Squared Gradient"),
        ("cumulative_squared_gradient_norm", "Cumulative Squared Gradient"),
        ("reference_point", "Reference Point"),
        ("cash_weight", "Cash Weight"),
        ("turnover", "Turnover"),
        ("disposition_spread", "Disposition Spread"),
    ]
    paths: list[Path] = []
    for method, method_data in merged.groupby("method", sort=True):
        paths.append(
            _plot_quantile_panels(
                method_data,
                output_dir / f"method_performance_{method}.png",
                panels,
                str(method),
                show_regime_boundaries=True,
            )
        )
    return paths


def save_synthetic_reference_comparison_plot(
    dynamic_trace: pd.DataFrame,
    static_trace: pd.DataFrame,
    output_path: Path,
    *,
    policy_label: str,
) -> Path:
    dynamic_trace = dynamic_trace.copy()
    static_trace = static_trace.copy()
    dynamic_trace["reference_mode"] = "dynamic"
    static_trace["reference_mode"] = "static"
    merged = pd.concat([dynamic_trace, static_trace], ignore_index=True)
    panels = [
        ("average_squared_gradient_norm", "Average Squared Gradient"),
        ("cumulative_squared_gradient_norm", "Cumulative Squared Gradient"),
        ("dynamic_local_regret", "Dynamic Local Regret"),
        ("objective_estimate", "CPT Objective Estimate"),
        ("normalized_wealth", "Normalized Wealth"),
        ("wealth", "Wealth"),
        ("relative_wealth", "Relative Wealth"),
        ("disposition_spread", "Disposition Spread"),
    ]
    return _plot_grouped_quantile_panels(merged, output_path, panels, "reference_mode", policy_label)


def _plot_quantile_panels(
    data: pd.DataFrame,
    path: Path,
    panels: list[tuple[str, str]],
    label: str,
    *,
    show_regime_boundaries: bool = False,
) -> Path:
    import matplotlib.pyplot as plt

    panels = [(column, title) for column, title in panels if column in data.columns]
    if not panels:
        raise RuntimeError("No requested columns are available for synthetic quantile plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.4 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (column, title) in zip(axes, panels):
        stats = _step_quantiles(data, column)
        ax.plot(stats["step"], stats["q50"], linewidth=1.35, label=label, color="#0072B2")
        ax.fill_between(stats["step"], stats["q25"], stats["q75"], alpha=0.22, color="#56B4E9", linewidth=0)
        if show_regime_boundaries:
            _add_regime_boundaries(ax, data)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Period")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _add_regime_boundaries(ax: object, data: pd.DataFrame) -> None:
    if "market_regime" not in data.columns:
        return
    regimes = (
        data[["step", "market_regime"]]
        .dropna()
        .drop_duplicates()
        .sort_values("step")
    )
    if regimes.empty:
        return
    spans: list[tuple[float, float, str]] = []
    start_step = float(regimes["step"].iloc[0])
    current_regime = str(regimes["market_regime"].iloc[0])
    previous_step = start_step
    for row in regimes.iloc[1:].itertuples(index=False):
        step = float(row.step)
        regime = str(row.market_regime)
        if regime != current_regime:
            spans.append((start_step, previous_step, current_regime))
            ax.axvline(step, color="#9CA3AF", linestyle="--", linewidth=0.8, alpha=0.65)
            start_step = step
            current_regime = regime
        previous_step = step
    spans.append((start_step, previous_step, current_regime))

    colors = ["#EFF6FF", "#F0FDF4", "#FEF2F2", "#FFFBEB"]
    for index, (left, right, regime) in enumerate(spans):
        ax.axvspan(left, right, color=colors[index % len(colors)], alpha=0.18, linewidth=0)
        midpoint = 0.5 * (left + right)
        ax.text(
            midpoint,
            0.98,
            regime,
            transform=ax.get_xaxis_transform(),
            rotation=0,
            va="top",
            ha="center",
            fontsize=7,
            color="#6B7280",
        )


def _plot_grouped_quantile_panels(
    data: pd.DataFrame,
    path: Path,
    panels: list[tuple[str, str]],
    group_column: str,
    title_prefix: str,
) -> Path:
    import matplotlib.pyplot as plt

    colors = {
        "dynamic": ("#D55E00", "#E69F00"),
        "static": ("#0072B2", "#56B4E9"),
        "dynamic_cpt_pg": ("#D55E00", "#E69F00"),
        "symmetric_cpt_pg": ("#CC79A7", "#F0A3C7"),
        "static_cpt_pg": ("#0072B2", "#56B4E9"),
        "expected_return_pg": ("#009E73", "#7AD7B1"),
        "exponential_utility_pg": ("#8237BB", "#D6B3FF"),
    }
    panels = [(column, title) for column, title in panels if column in data.columns]
    if not panels:
        raise RuntimeError("No requested columns are available for synthetic grouped quantile plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.4 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (column, title) in zip(axes, panels):
        for group_value, group_data in data.groupby(group_column):
            line_color, fill_color = colors.get(str(group_value), ("#000000", "#999999"))
            stats = _step_quantiles(group_data, column)
            ax.plot(stats["step"], stats["q50"], linewidth=1.35, label=str(group_value), color=line_color)
            ax.fill_between(stats["step"], stats["q25"], stats["q75"], alpha=0.20, color=fill_color, linewidth=0)
        ax.set_title(f"{title_prefix}: {title}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Period")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _step_quantiles(data: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = data.groupby("step", sort=True)[column]
    return pd.DataFrame(
        {
            "step": grouped.median().index.to_numpy(dtype=float),
            "q25": grouped.quantile(0.25).to_numpy(dtype=float),
            "q50": grouped.median().to_numpy(dtype=float),
            "q75": grouped.quantile(0.75).to_numpy(dtype=float),
        }
    )


def disposition_effect_metrics(
    previous_weights: np.ndarray,
    target_weights: np.ndarray,
    open_prices: np.ndarray,
    cost_basis: np.ndarray,
) -> dict[str, float]:
    previous = np.asarray(previous_weights, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    prices = np.asarray(open_prices, dtype=float)
    basis = np.asarray(cost_basis, dtype=float)
    if previous.shape != target.shape or previous.shape != prices.shape or previous.shape != basis.shape:
        raise RuntimeError("Disposition effect metric inputs must have the same shape")
    risky_mask = np.arange(len(previous)) != 0
    held_mask = risky_mask & (previous > 1e-12) & np.isfinite(basis)
    safe_prices = np.maximum(prices, 1e-12)
    previous_shares = previous / safe_prices
    target_shares = target / safe_prices
    sold_mask = held_mask & (target_shares < previous_shares - 1e-12)
    paper_mask = held_mask & ~sold_mask
    gain_mask = held_mask & (prices > basis + 1e-12)
    loss_mask = held_mask & (prices < basis - 1e-12)
    realized_gain_count = int(np.count_nonzero(sold_mask & gain_mask))
    realized_loss_count = int(np.count_nonzero(sold_mask & loss_mask))
    paper_gain_count = int(np.count_nonzero(paper_mask & gain_mask))
    paper_loss_count = int(np.count_nonzero(paper_mask & loss_mask))
    gain_denominator = realized_gain_count + paper_gain_count
    loss_denominator = realized_loss_count + paper_loss_count
    pgr = 0.0 if gain_denominator == 0 else float(realized_gain_count / gain_denominator)
    plr = 0.0 if loss_denominator == 0 else float(realized_loss_count / loss_denominator)
    return {
        "pgr": pgr,
        "plr": plr,
        "disposition_spread": pgr - plr,
        "realized_gain_count": realized_gain_count,
        "realized_loss_count": realized_loss_count,
        "paper_gain_count": paper_gain_count,
        "paper_loss_count": paper_loss_count,
    }
