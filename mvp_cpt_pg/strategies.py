from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .actions import (
    CASH_CODE,
    POLICY_FEATURE_COLUMNS,
    build_continuous_policy_state,
    check_constraint_violation,
    constraint_violation_reason,
    continuous_action_summary,
    project_continuous_weights_array,
    project_continuous_weights,
)
from .config import ExperimentConfig
from .market_data import MarketDataset
from .schemas import HardConstraints, PreferenceVector
from .utils import normalize_weights


@dataclass
class PortfolioDecision:
    action_name: str
    weights: pd.Series
    metadata: dict[str, Any]


@dataclass
class StrategyStep:
    day_return: float
    day_return_rate: float
    investment_return_rate: float
    day_relative_return_rate: float
    portfolio_value: float
    transaction_cost: float
    traded_value: float
    objective_estimate: float
    offline_cpt_common_ref: float
    gradient_norm: float
    gradient_bootstrap_error_norm: float
    gradient_bootstrap_std_norm: float
    gradient_bootstrap_se_norm: float
    gradient_bootstrap_relative_error: float
    objective_bootstrap_std: float
    gradient_diagnostic_repeats: int
    cpt_sample_count: int
    gradient_sample_count: int
    update_norm: float
    theta_norm: float
    theta_max_abs: float
    theta_boundary_share: float
    gradient_vector: list[float]
    theta_before_vector: list[float]
    theta_after_vector: list[float]
    update_vector: list[float]
    reference_drift: float
    turnover: float
    constraint_violation: int
    constraint_violation_reason: str


@dataclass(frozen=True)
class PortfolioOutcome:
    end_value: float
    day_pnl: float
    day_return_rate: float
    investment_return_rate: float
    invested_value: float
    transaction_cost: float
    traded_value: float
    turnover: float


class BaseStrategy:
    def __init__(self, name: str, config: ExperimentConfig, dataset: MarketDataset, seed: int) -> None:
        self.name = name
        self.config = config
        self.dataset = dataset
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.return_matrix = dataset.panel.pivot(index="trade_date", columns="ts_code", values="open_close_ret").sort_index().fillna(0.0)
        self.universe_codes = dataset.universe["ts_code"].tolist()
        self.asset_codes = [*self.universe_codes, CASH_CODE]
        self.previous_weights = pd.Series(0.0, index=self.asset_codes, dtype=float)
        self.previous_weights.loc[CASH_CODE] = 1.0
        self.portfolio_value = float(config.initial_capital_amount)
        self.budget_limit = float(config.initial_capital_amount)
        self.performance_base = float(config.initial_capital_amount)
        self.reference_point = 0.0
        self.seen_dates: list[str] = []
        self.last_gradient_norm = 0.0
        self.last_gradient_bootstrap_error_norm = math.nan
        self.last_gradient_bootstrap_std_norm = math.nan
        self.last_gradient_bootstrap_se_norm = math.nan
        self.last_gradient_bootstrap_relative_error = math.nan
        self.last_objective_bootstrap_std = math.nan
        self.last_gradient_diagnostic_repeats = 1
        self.last_cpt_sample_count = 0
        self.last_gradient_sample_count = 0
        self.last_update_norm = 0.0
        self.last_theta_norm = math.nan
        self.last_theta_max_abs = math.nan
        self.last_theta_boundary_share = math.nan
        self.last_objective_estimate = math.nan
        self.last_offline_cpt_common_ref = math.nan
        self.last_gradient_vector: list[float] = []
        self.last_theta_before_vector: list[float] = []
        self.last_theta_after_vector: list[float] = []
        self.last_update_vector: list[float] = []

    @property
    def use_cpt_optimizer(self) -> bool:
        return False

    @property
    def static_reference(self) -> bool:
        return False

    @property
    def frozen_preference(self) -> bool:
        return False

    def initialize(self) -> None:
        return None

    def effective_preference(self, preference: PreferenceVector) -> PreferenceVector:
        return preference

    def objective_reference(self) -> float:
        return self.reference_point

    def set_initial_portfolio(
        self,
        reference_point: float,
        weights: pd.Series | None = None,
        portfolio_value: float | None = None,
    ) -> None:
        self.reference_point = float(reference_point)
        if weights is not None:
            self.previous_weights = weights.reindex(self.asset_codes, fill_value=0.0)
        if portfolio_value is not None:
            self.portfolio_value = float(portfolio_value)
            self.performance_base = float(portfolio_value)

    def trade_capital(self) -> float:
        return float(self.portfolio_value)

    def select(self, trade_date: str, preference: PreferenceVector, hard_constraints: HardConstraints) -> PortfolioDecision:
        raise NotImplementedError

    def after_day(
        self,
        trade_date: str,
        weights: pd.Series,
        outcome: PortfolioOutcome,
        preference: PreferenceVector,
        hard_constraints: HardConstraints,
    ) -> StrategyStep:
        objective_reference = self.objective_reference()
        previous_reference = self.reference_point
        previous_weights = self.previous_weights.copy()
        self.seen_dates.append(trade_date)
        objective_signal = outcome.investment_return_rate
        if not self.static_reference:
            self.reference_point = update_reference_point(
                self.reference_point,
                objective_signal,
                self.config.eta_gain,
                self.config.eta_loss,
            )
        reference_drift = abs(self.reference_point - previous_reference)
        self.previous_weights = weights.reindex(self.asset_codes, fill_value=0.0)
        self.portfolio_value = outcome.end_value
        return StrategyStep(
            day_return=outcome.day_pnl,
            day_return_rate=outcome.day_return_rate,
            investment_return_rate=outcome.investment_return_rate,
            day_relative_return_rate=outcome.investment_return_rate - objective_reference,
            portfolio_value=outcome.end_value,
            transaction_cost=outcome.transaction_cost,
            traded_value=outcome.traded_value,
            objective_estimate=self.last_objective_estimate,
            offline_cpt_common_ref=self.last_offline_cpt_common_ref,
            gradient_norm=self.last_gradient_norm,
            gradient_bootstrap_error_norm=self.last_gradient_bootstrap_error_norm,
            gradient_bootstrap_std_norm=self.last_gradient_bootstrap_std_norm,
            gradient_bootstrap_se_norm=self.last_gradient_bootstrap_se_norm,
            gradient_bootstrap_relative_error=self.last_gradient_bootstrap_relative_error,
            objective_bootstrap_std=self.last_objective_bootstrap_std,
            gradient_diagnostic_repeats=self.last_gradient_diagnostic_repeats,
            cpt_sample_count=self.last_cpt_sample_count,
            gradient_sample_count=self.last_gradient_sample_count,
            update_norm=self.last_update_norm,
            theta_norm=self.last_theta_norm,
            theta_max_abs=self.last_theta_max_abs,
            theta_boundary_share=self.last_theta_boundary_share,
            gradient_vector=self.last_gradient_vector,
            theta_before_vector=self.last_theta_before_vector,
            theta_after_vector=self.last_theta_after_vector,
            update_vector=self.last_update_vector,
            reference_drift=reference_drift,
            turnover=outcome.turnover,
            constraint_violation=check_constraint_violation(weights, hard_constraints, previous_weights),
            constraint_violation_reason=constraint_violation_reason(weights, hard_constraints, previous_weights),
        )

    def _update_model(self, trade_date: str, preference: PreferenceVector, hard_constraints: HardConstraints) -> None:
        self.last_gradient_norm = 0.0
        self.last_gradient_bootstrap_error_norm = math.nan
        self.last_gradient_bootstrap_std_norm = math.nan
        self.last_gradient_bootstrap_se_norm = math.nan
        self.last_gradient_bootstrap_relative_error = math.nan
        self.last_objective_bootstrap_std = math.nan
        self.last_gradient_diagnostic_repeats = 1
        self.last_cpt_sample_count = 0
        self.last_gradient_sample_count = 0
        self.last_update_norm = 0.0
        self.last_theta_norm = math.nan
        self.last_theta_max_abs = math.nan
        self.last_theta_boundary_share = math.nan
        self.last_objective_estimate = math.nan
        self.last_offline_cpt_common_ref = math.nan
        self.last_gradient_vector = []
        self.last_theta_before_vector = []
        self.last_theta_after_vector = []
        self.last_update_vector = []


class CPTPGStrategy(BaseStrategy):
    def __init__(
        self,
        name: str,
        config: ExperimentConfig,
        dataset: MarketDataset,
        seed: int,
        *,
        static_reference: bool = False,
        frozen_preference: bool = False,
    ) -> None:
        super().__init__(name=name, config=config, dataset=dataset, seed=seed)
        self.theta = np.zeros(len(POLICY_FEATURE_COLUMNS), dtype=float)
        self.gamma0 = config.gamma0
        self.policy_noise_scale = config.policy_noise_scale
        self._static_reference = static_reference
        self._frozen_preference = frozen_preference
        self._last_update_trade_date: str | None = None

    @property
    def use_cpt_optimizer(self) -> bool:
        return True

    @property
    def static_reference(self) -> bool:
        return self._static_reference

    @property
    def frozen_preference(self) -> bool:
        return self._frozen_preference

    def select(self, trade_date: str, preference: PreferenceVector, hard_constraints: HardConstraints) -> PortfolioDecision:
        pref = self.effective_preference(preference)
        self._update_model(trade_date, pref, hard_constraints)
        observed_state = self.dataset.observed_stock_state(trade_date)
        policy_state = build_continuous_policy_state(
            observed_state,
            pref,
            prev_weights=self.previous_weights,
        )
        weights, _, metadata = self._sample_continuous_action(policy_state, hard_constraints, self.previous_weights)
        return PortfolioDecision(
            action_name="continuous_weight_policy",
            weights=weights,
            metadata=metadata,
        )

    def _sample_continuous_action(
        self,
        policy_state,
        hard_constraints: HardConstraints,
        prev_weights: pd.Series,
        rng: np.random.Generator | None = None,
    ) -> tuple[pd.Series, np.ndarray, dict[str, Any]]:
        if self.policy_noise_scale <= 0:
            raise RuntimeError("policy_noise_scale must be positive")
        active_rng = rng if rng is not None else self.rng
        mean_vector = policy_state.feature_matrix @ self.theta
        noise = active_rng.normal(loc=0.0, scale=self.policy_noise_scale, size=len(policy_state.codes))
        latent = mean_vector + noise
        raw_risky_weights = policy_weights_from_latent(policy_state.codes, latent, self.config.policy_normalizer)
        projection_codes = [*policy_state.codes, CASH_CODE]
        raw_weights = raw_risky_weights.reindex(projection_codes, fill_value=0.0)
        weights = project_continuous_weights(
            codes=projection_codes,
            raw_weights=raw_weights,
            hard_constraints=hard_constraints,
            prev_weights=prev_weights.reindex(projection_codes, fill_value=0.0),
        )
        latent_score = (latent - mean_vector) / (self.policy_noise_scale**2)
        score_gradient = policy_state.feature_matrix.T @ latent_score
        action_summary = continuous_action_summary(
            weights=weights,
            prev_weights=prev_weights,
            portfolio_value=self.portfolio_value,
            news_context=policy_state.news_context,
        )
        metadata = {
            "policy_mean_vector": mean_vector,
            "action_summary": action_summary,
        }
        return weights, score_gradient, metadata

    def _sample_continuous_action_batch(
        self,
        policy_state,
        hard_constraints: HardConstraints,
        prev_weight_matrix: np.ndarray,
        prev_codes: list[str],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.policy_noise_scale <= 0:
            raise RuntimeError("policy_noise_scale must be positive")
        mean_vector = policy_state.feature_matrix @ self.theta
        noise = rng.normal(
            loc=0.0,
            scale=self.policy_noise_scale,
            size=(prev_weight_matrix.shape[0], len(policy_state.codes)),
        )
        latent_matrix = mean_vector.reshape(1, -1) + noise
        raw_risky_weight_matrix = policy_weight_matrix_from_latent(
            policy_state.codes,
            latent_matrix,
            self.config.policy_normalizer,
        )
        projection_codes = [*policy_state.codes, CASH_CODE]
        aligned_prev = align_weight_matrix(prev_weight_matrix, prev_codes, projection_codes)
        raw_weight_matrix = np.zeros((raw_risky_weight_matrix.shape[0], len(projection_codes)), dtype=float)
        raw_weight_matrix[:, : len(policy_state.codes)] = raw_risky_weight_matrix
        projected_rows = []
        for row_index, raw_weights in enumerate(raw_weight_matrix):
            projected = project_continuous_weights_array(
                codes=projection_codes,
                raw_weights=raw_weights,
                hard_constraints=hard_constraints,
                prev_weights=aligned_prev[row_index],
            )
            projected_rows.append(projected)
        projected_weight_matrix = np.vstack(projected_rows)
        latent_score_matrix = (latent_matrix - mean_vector.reshape(1, -1)) / (self.policy_noise_scale**2)
        score_gradient_matrix = latent_score_matrix @ policy_state.feature_matrix
        return projected_weight_matrix, score_gradient_matrix

    def _update_model(self, trade_date: str, preference: PreferenceVector, hard_constraints: HardConstraints) -> None:
        if self._last_update_trade_date == trade_date:
            return
        self._last_update_trade_date = trade_date
        history_dates = [date for date in self.dataset.trade_dates if date < trade_date][-self.config.evaluation_horizon :]
        if len(history_dates) < self.config.evaluation_horizon:
            raise RuntimeError(
                f"Insufficient history before {trade_date}: expected {self.config.evaluation_horizon} trade dates, got {len(history_dates)}"
            )
        pref = self.effective_preference(preference)
        repeat_count = max(1, int(self.config.gradient_diagnostic_repeats))
        gradients: list[np.ndarray] = []
        objective_estimates: list[float] = []
        offline_cpt_common_refs: list[float] = []
        cpt_sample_counts: list[int] = []
        gradient_sample_counts: list[int] = []
        for repeat_index in range(repeat_count):
            gradient_i, objective_i, offline_i, cpt_n_i, gradient_m_i = self._estimate_gradient_and_objective(
                history_dates,
                pref,
                hard_constraints,
                cpt_rng=self._sampling_rng(trade_date, repeat_index, "cpt"),
                gradient_rng=self._sampling_rng(trade_date, repeat_index, "gradient"),
            )
            gradients.append(gradient_i)
            objective_estimates.append(float(objective_i))
            offline_cpt_common_refs.append(float(offline_i))
            cpt_sample_counts.append(int(cpt_n_i))
            gradient_sample_counts.append(int(gradient_m_i))
        gradient_stack = np.vstack(gradients)
        mean_gradient = gradient_stack.mean(axis=0)
        if repeat_count > 1 and self.config.gradient_diagnostic_use_mean_update:
            gradient = mean_gradient
            objective_estimate = float(np.mean(objective_estimates))
            offline_cpt_common_ref = float(np.mean(offline_cpt_common_refs))
        else:
            gradient = gradients[0]
            objective_estimate = float(objective_estimates[0])
            offline_cpt_common_ref = float(offline_cpt_common_refs[0])
        if repeat_count > 1:
            centered = gradient_stack - mean_gradient
            gradient_bootstrap_error_norm = float(np.linalg.norm(centered, axis=1).mean())
            gradient_bootstrap_std_norm = float(np.linalg.norm(gradient_stack.std(axis=0, ddof=0)))
            gradient_bootstrap_se_norm = float(gradient_bootstrap_std_norm / math.sqrt(repeat_count))
            gradient_bootstrap_relative_error = float(gradient_bootstrap_error_norm / (np.linalg.norm(mean_gradient) + 1e-12))
            objective_bootstrap_std = float(np.std(objective_estimates, ddof=0))
        else:
            gradient_bootstrap_error_norm = math.nan
            gradient_bootstrap_std_norm = math.nan
            gradient_bootstrap_se_norm = math.nan
            gradient_bootstrap_relative_error = math.nan
            objective_bootstrap_std = math.nan
        gradient_norm = float(np.linalg.norm(gradient))
        step_index = max(1, len(self.seen_dates) + 1)
        gamma_t = self.gamma0 / (step_index ** self.config.gamma_exponent)
        update = gamma_t * gradient
        theta_before = self.theta.copy()
        theta_after = np.clip(theta_before + update, -self.config.max_logit_abs, self.config.max_logit_abs)
        self.theta = theta_after
        self.last_gradient_norm = gradient_norm
        self.last_gradient_bootstrap_error_norm = gradient_bootstrap_error_norm
        self.last_gradient_bootstrap_std_norm = gradient_bootstrap_std_norm
        self.last_gradient_bootstrap_se_norm = gradient_bootstrap_se_norm
        self.last_gradient_bootstrap_relative_error = gradient_bootstrap_relative_error
        self.last_objective_bootstrap_std = objective_bootstrap_std
        self.last_gradient_diagnostic_repeats = repeat_count
        self.last_cpt_sample_count = int(cpt_sample_counts[0]) if cpt_sample_counts else 0
        self.last_gradient_sample_count = int(gradient_sample_counts[0]) if gradient_sample_counts else 0
        self.last_update_norm = float(np.linalg.norm(update))
        self.last_theta_norm = float(np.linalg.norm(theta_after))
        self.last_theta_max_abs = float(np.max(np.abs(theta_after)))
        self.last_theta_boundary_share = float(np.mean(np.abs(theta_after) >= self.config.max_logit_abs - 1e-8))
        self.last_objective_estimate = float(objective_estimate)
        self.last_offline_cpt_common_ref = float(offline_cpt_common_ref)
        self.last_gradient_vector = gradient.astype(float).tolist()
        self.last_theta_before_vector = theta_before.astype(float).tolist()
        self.last_theta_after_vector = theta_after.astype(float).tolist()
        self.last_update_vector = update.astype(float).tolist()

    def _sampling_rng(self, trade_date: str, repeat_index: int, sample_group: str) -> np.random.Generator:
        key = f"{self.name}|{self.seed}|{trade_date}|{repeat_index}|{sample_group}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        return np.random.default_rng(seed)

    def _estimate_gradient_and_objective(
        self,
        history_dates: list[str],
        preference: PreferenceVector,
        hard_constraints: HardConstraints,
        cpt_rng: np.random.Generator,
        gradient_rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, float, int, int]:
        online_step = max(1, len(self.seen_dates) + 1)
        n_t = self._sample_count(self.config.cpt_sample_base, online_step)
        m_t = self._sample_count(self.config.gradient_sample_base, online_step)
        objective_reference = self.objective_reference()
        cpt_returns, _ = self._sample_window_outcomes(
            history_dates=history_dates,
            preference=preference,
            hard_constraints=hard_constraints,
            sample_count=n_t,
            rng=cpt_rng,
            collect_scores=False,
        )
        gradient_returns, score_matrix = self._sample_window_outcomes(
            history_dates=history_dates,
            preference=preference,
            hard_constraints=hard_constraints,
            sample_count=m_t,
            rng=gradient_rng,
            collect_scores=True,
        )
        gradient_relative_returns = gradient_returns - objective_reference
        common_reference_returns = cpt_returns - self.config.offline_cpt_reference
        if score_matrix is None:
            raise RuntimeError("Gradient sample scores were not collected")
        gradient_score_vectors = [score_matrix[row_index] for row_index in range(m_t)]
        gradient = compute_cpt_gradient(gradient_relative_returns.tolist(), gradient_score_vectors, self.config)
        return (
            gradient,
            self._objective_from_returns(cpt_returns, objective_reference),
            compute_cpt_objective(common_reference_returns.tolist(), self.config),
            n_t,
            m_t,
        )

    def _objective_from_returns(self, returns: np.ndarray, objective_reference: float) -> float:
        relative_returns = returns - objective_reference
        return compute_cpt_objective(relative_returns.tolist(), self.config)

    def _sample_count(self, base_count: int, online_step: int) -> int:
        if self.config.fixed_sample_counts:
            return max(1, int(base_count))
        return max(1, int(math.ceil(base_count * (online_step ** self.config.sample_exponent))))

    def _sample_window_outcomes(
        self,
        *,
        history_dates: list[str],
        preference: PreferenceVector,
        hard_constraints: HardConstraints,
        sample_count: int,
        rng: np.random.Generator,
        collect_scores: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        starting_value = self.trade_capital()
        simulated_values = np.full(sample_count, starting_value, dtype=float)
        trajectory_growth = np.ones(sample_count, dtype=float)
        score_matrix = np.zeros((sample_count, len(POLICY_FEATURE_COLUMNS)), dtype=float) if collect_scores else None
        prev_codes = list(self.asset_codes)
        prev_weight_matrix = np.tile(
            self.previous_weights.reindex(prev_codes, fill_value=0.0).to_numpy(dtype=float),
            (sample_count, 1),
        )
        for history_date in history_dates:
            observed_state = self.dataset.observed_stock_state(history_date)
            policy_state = build_continuous_policy_state(
                observed_state,
                preference,
                prev_weights=pd.Series(prev_weight_matrix.mean(axis=0), index=prev_codes, dtype=float),
            )
            weight_matrix, score_gradient_matrix = self._sample_continuous_action_batch(
                policy_state,
                hard_constraints,
                prev_weight_matrix,
                prev_codes,
                rng,
            )
            if score_matrix is not None:
                score_matrix += score_gradient_matrix
            projection_codes = [*policy_state.codes, CASH_CODE]
            aligned_prev = align_weight_matrix(prev_weight_matrix, prev_codes, projection_codes)
            returns = self.return_matrix.loc[history_date].reindex(projection_codes, fill_value=0.0).fillna(0.0)
            returns.loc[CASH_CODE] = 0.0
            day_returns = returns.to_numpy(dtype=float)
            risky_columns = np.array([code != CASH_CODE for code in projection_codes], dtype=bool)
            risky_weight_sums = weight_matrix[:, risky_columns].sum(axis=1)
            invested_values = simulated_values * risky_weight_sums
            turnover = np.abs(weight_matrix[:, risky_columns] - aligned_prev[:, risky_columns]).sum(axis=1)
            transaction_cost = simulated_values * turnover * self.config.trade_cost_bps / 10000.0
            investable_values = np.maximum(simulated_values - transaction_cost, 0.0)
            weighted_returns = weight_matrix @ day_returns
            previous_values = simulated_values
            simulated_values = investable_values * (1.0 + weighted_returns)
            day_net_pnl = simulated_values - previous_values
            day_investment_return = np.divide(
                day_net_pnl,
                invested_values,
                out=np.zeros_like(day_net_pnl),
                where=invested_values > 1e-12,
            )
            trajectory_growth *= 1.0 + day_investment_return
            prev_codes = projection_codes
            prev_weight_matrix = weight_matrix
        investment_return = trajectory_growth - 1.0
        return investment_return, score_matrix


class ExpectedReturnPGStrategy(CPTPGStrategy):
    def _objective_from_returns(self, returns: np.ndarray, objective_reference: float) -> float:
        return float(np.mean(returns))

    def _estimate_gradient_and_objective(
        self,
        history_dates: list[str],
        preference: PreferenceVector,
        hard_constraints: HardConstraints,
        cpt_rng: np.random.Generator,
        gradient_rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, float, int, int]:
        online_step = max(1, len(self.seen_dates) + 1)
        n_t = self._sample_count(self.config.cpt_sample_base, online_step)
        m_t = self._sample_count(self.config.gradient_sample_base, online_step)
        objective_returns, _ = self._sample_window_outcomes(
            history_dates=history_dates,
            preference=preference,
            hard_constraints=hard_constraints,
            sample_count=n_t,
            rng=cpt_rng,
            collect_scores=False,
        )
        gradient_returns, score_matrix = self._sample_window_outcomes(
            history_dates=history_dates,
            preference=preference,
            hard_constraints=hard_constraints,
            sample_count=m_t,
            rng=gradient_rng,
            collect_scores=True,
        )
        if score_matrix is None:
            raise RuntimeError("Gradient sample scores were not collected")
        common_reference_returns = objective_returns - self.config.offline_cpt_reference
        return (
            compute_expected_return_gradient(gradient_returns, score_matrix),
            float(np.mean(objective_returns)),
            compute_cpt_objective(common_reference_returns.tolist(), self.config),
            n_t,
            m_t,
        )


class ExponentialUtilityPGStrategy(CPTPGStrategy):
    def _objective_from_returns(self, returns: np.ndarray, objective_reference: float) -> float:
        return compute_exponential_utility_objective(
            returns,
            risk_aversion=self.config.exponential_risk_aversion,
        )

    def _estimate_gradient_and_objective(
        self,
        history_dates: list[str],
        preference: PreferenceVector,
        hard_constraints: HardConstraints,
        cpt_rng: np.random.Generator,
        gradient_rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, float, int, int]:
        online_step = max(1, len(self.seen_dates) + 1)
        n_t = self._sample_count(self.config.cpt_sample_base, online_step)
        m_t = self._sample_count(self.config.gradient_sample_base, online_step)
        objective_returns, _ = self._sample_window_outcomes(
            history_dates=history_dates,
            preference=preference,
            hard_constraints=hard_constraints,
            sample_count=n_t,
            rng=cpt_rng,
            collect_scores=False,
        )
        gradient_returns, score_matrix = self._sample_window_outcomes(
            history_dates=history_dates,
            preference=preference,
            hard_constraints=hard_constraints,
            sample_count=m_t,
            rng=gradient_rng,
            collect_scores=True,
        )
        if score_matrix is None:
            raise RuntimeError("Gradient sample scores were not collected")
        common_reference_returns = objective_returns - self.config.offline_cpt_reference
        return (
            compute_exponential_utility_gradient(
                gradient_returns,
                score_matrix,
                risk_aversion=self.config.exponential_risk_aversion,
            ),
            compute_exponential_utility_objective(
                objective_returns,
                risk_aversion=self.config.exponential_risk_aversion,
            ),
            compute_cpt_objective(common_reference_returns.tolist(), self.config),
            n_t,
            m_t,
        )


def build_strategy(method: str, config: ExperimentConfig, dataset: MarketDataset, seed: int) -> BaseStrategy:
    if method == "dynamic_cpt_pg":
        return CPTPGStrategy(method, config, dataset, seed, static_reference=False, frozen_preference=False)
    if method == "static_cpt_pg":
        return CPTPGStrategy(method, config, dataset, seed, static_reference=True, frozen_preference=True)
    if method == "dynamic_cpt_pg_frozen_pref":
        return CPTPGStrategy(method, config, dataset, seed, static_reference=False, frozen_preference=True)
    if method == "static_ref_dynamic_pref_cpt_pg":
        return CPTPGStrategy(method, config, dataset, seed, static_reference=True, frozen_preference=False)
    if method == "expected_return_pg":
        return ExpectedReturnPGStrategy(method, config, dataset, seed, static_reference=False, frozen_preference=False)
    if method == "exponential_utility_pg":
        return ExponentialUtilityPGStrategy(method, config, dataset, seed, static_reference=False, frozen_preference=False)
    raise ValueError(f"Unknown method: {method}")


def softmax_vector(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp_vals = np.exp(np.clip(shifted, -30.0, 30.0))
    denom = float(exp_vals.sum())
    if denom <= 0:
        raise RuntimeError("Continuous policy softmax normalization failed")
    return exp_vals / denom


def policy_weights_from_latent(codes: list[str], latent: np.ndarray, normalizer: str = "softmax") -> pd.Series:
    if len(codes) != len(latent):
        raise RuntimeError("Policy codes and latent vector length mismatch")
    weights = normalize_latent_vector(np.asarray(latent, dtype=float), normalizer)
    return pd.Series(weights, index=codes, dtype=float)


def policy_weight_matrix_from_latent(codes: list[str], latent_matrix: np.ndarray, normalizer: str = "softmax") -> np.ndarray:
    if latent_matrix.ndim != 2 or latent_matrix.shape[1] != len(codes):
        raise RuntimeError("Policy codes and latent matrix shape mismatch")
    return normalize_latent_matrix(latent_matrix, normalizer)


def normalize_latent_vector(values: np.ndarray, normalizer: str) -> np.ndarray:
    if normalizer == "softmax":
        return softmax_vector(values)
    if normalizer == "sparsemax":
        return sparsemax_vector(values)
    raise ValueError(f"Unknown policy_normalizer: {normalizer}")


def normalize_latent_matrix(values: np.ndarray, normalizer: str) -> np.ndarray:
    if normalizer == "softmax":
        return softmax_matrix(values)
    if normalizer == "sparsemax":
        return sparsemax_matrix(values)
    raise ValueError(f"Unknown policy_normalizer: {normalizer}")


def softmax_matrix(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denominator = exp_values.sum(axis=1, keepdims=True)
    denominator = np.where(denominator <= 0.0, 1.0, denominator)
    return exp_values / denominator


def sparsemax_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise RuntimeError("Continuous policy sparsemax expects a non-empty vector")
    shifted = values - float(np.mean(values))
    sorted_values = np.sort(shifted)[::-1]
    cssv = np.cumsum(sorted_values)
    ks = np.arange(1, len(sorted_values) + 1, dtype=float)
    support = 1.0 + ks * sorted_values > cssv
    if not bool(support.any()):
        return np.full(len(values), 1.0 / len(values), dtype=float)
    k_z = int(ks[support][-1])
    tau = float((cssv[k_z - 1] - 1.0) / k_z)
    weights = np.maximum(shifted - tau, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(len(values), 1.0 / len(values), dtype=float)
    return weights / total


def sparsemax_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise RuntimeError("Continuous policy sparsemax expects a matrix")
    if values.shape[0] == 0:
        return values.copy()
    shifted = values - np.mean(values, axis=1, keepdims=True)
    sorted_values = -np.sort(-shifted, axis=1)
    cssv = np.cumsum(sorted_values, axis=1)
    ks = np.arange(1, values.shape[1] + 1, dtype=float).reshape(1, -1)
    support = 1.0 + ks * sorted_values > cssv
    support_count = np.maximum(support.sum(axis=1), 1)
    tau = (cssv[np.arange(values.shape[0]), support_count - 1] - 1.0) / support_count
    weights = np.maximum(shifted - tau.reshape(-1, 1), 0.0)
    totals = weights.sum(axis=1, keepdims=True)
    fallback = np.full_like(weights, 1.0 / values.shape[1])
    return np.where(totals > 0.0, weights / np.where(totals <= 0.0, 1.0, totals), fallback)


def align_weight_matrix(weight_matrix: np.ndarray, source_codes: list[str], target_codes: list[str]) -> np.ndarray:
    if source_codes == target_codes:
        return weight_matrix
    frame = pd.DataFrame(weight_matrix, columns=source_codes)
    return frame.reindex(columns=target_codes, fill_value=0.0).to_numpy(dtype=float)


def traded_notional(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    all_codes = sorted(code for code in set(prev_weights.index).union(new_weights.index) if code != CASH_CODE)
    prev = prev_weights.reindex(all_codes, fill_value=0.0)
    new = new_weights.reindex(all_codes, fill_value=0.0)
    return float((new - prev).abs().sum())


def portfolio_step_value(
    day_returns: pd.Series,
    prev_weights: pd.Series,
    weights: pd.Series,
    portfolio_value: float,
    trade_cost_bps: float,
) -> PortfolioOutcome:
    weights = normalize_weights(weights)
    current_value = max(float(portfolio_value), 0.0)
    turnover = traded_notional(prev_weights, weights)
    traded_value = current_value * turnover
    transaction_cost = traded_value * trade_cost_bps / 10000.0
    investable_value = max(current_value - transaction_cost, 0.0)
    stock_weights = weights.drop(labels=[CASH_CODE], errors="ignore")
    risky_weight_sum = float(stock_weights.clip(lower=0.0).sum())
    invested_value = current_value * risky_weight_sum
    weighted_return = float((stock_weights.reindex(day_returns.index, fill_value=0.0) * day_returns.fillna(0.0)).sum())
    end_value = investable_value * (1.0 + weighted_return)
    day_pnl = end_value - current_value
    day_return_rate = 0.0 if current_value <= 0 else day_pnl / current_value
    investment_return_rate = 0.0 if invested_value <= 1e-12 else day_pnl / invested_value
    return PortfolioOutcome(
        end_value=end_value,
        day_pnl=day_pnl,
        day_return_rate=day_return_rate,
        investment_return_rate=investment_return_rate,
        invested_value=invested_value,
        transaction_cost=transaction_cost,
        traded_value=traded_value,
        turnover=turnover,
    )


def update_reference_point(current_reference: float, signal: float, eta_gain: float, eta_loss: float) -> float:
    if signal >= current_reference:
        return current_reference + eta_gain * (signal - current_reference)
    return current_reference - eta_loss * (current_reference - signal)


def gain_value(relative_return: float, config: ExperimentConfig) -> float:
    gain_magnitude = max(relative_return, 0.0)
    return smoothed_power_value(gain_magnitude, config.alpha_gain, config.cpt_value_smoothing)


def loss_value(relative_return: float, config: ExperimentConfig) -> float:
    loss_magnitude = max(-relative_return, 0.0)
    return float(-config.loss_aversion * smoothed_power_value(loss_magnitude, config.alpha_loss, config.cpt_value_smoothing))


def smoothed_power_value(magnitude: float, exponent: float, smoothing: float) -> float:
    x = max(float(magnitude), 0.0)
    eps = max(float(smoothing), 0.0)
    if eps <= 0.0:
        return float(x**exponent)
    return float(((x * x + eps * eps) ** (0.5 * exponent)) - (eps**exponent))


def cpt_probability_weight(probability: float, beta: float) -> float:
    p = min(1.0 - 1e-8, max(1e-8, probability))
    numerator = p**beta
    denominator = (p**beta + (1.0 - p) ** beta) ** (1.0 / beta)
    return float(numerator / denominator)


def compute_cpt_gradient(
    relative_returns: list[float],
    score_vectors: list[np.ndarray],
    config: ExperimentConfig,
) -> np.ndarray:
    if not relative_returns:
        return np.zeros(len(POLICY_FEATURE_COLUMNS), dtype=float)
    order = np.argsort(relative_returns)
    sorted_returns = [relative_returns[idx] for idx in order]
    sorted_scores = [score_vectors[idx] for idx in order]
    m = len(sorted_returns)
    loss_count = sum(value < 0 for value in sorted_returns)
    gradient = np.zeros_like(sorted_scores[0], dtype=float)
    for position, (rel, score) in enumerate(zip(sorted_returns, sorted_scores), start=1):
        if position <= loss_count:
            delta = cpt_probability_weight(position / m, config.beta_loss) - cpt_probability_weight((position - 1) / m, config.beta_loss)
            psi = loss_value(rel, config) * delta
        else:
            upper = (m - position + 1) / m
            lower = (m - position) / m
            delta = cpt_probability_weight(upper, config.beta_gain) - cpt_probability_weight(lower, config.beta_gain)
            psi = gain_value(rel, config) * delta
        gradient += psi * score
    gradient /= float(m)
    return gradient


def compute_expected_return_gradient(return_samples: np.ndarray, score_matrix: np.ndarray) -> np.ndarray:
    if len(return_samples) == 0:
        return np.zeros(len(POLICY_FEATURE_COLUMNS), dtype=float)
    if score_matrix.shape[0] != len(return_samples):
        raise RuntimeError("Expected-return gradient samples and score vectors length mismatch")
    return np.mean(return_samples.reshape(-1, 1) * score_matrix, axis=0)


def compute_exponential_utility_gradient(
    return_samples: np.ndarray,
    score_matrix: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    if len(return_samples) == 0:
        return np.zeros(len(POLICY_FEATURE_COLUMNS), dtype=float)
    if score_matrix.shape[0] != len(return_samples):
        raise RuntimeError("Exponential utility gradient samples and score vectors length mismatch")
    utility = exponential_utility_values(return_samples, risk_aversion)
    return np.mean(utility.reshape(-1, 1) * score_matrix, axis=0)


def compute_exponential_utility_objective(
    return_samples: np.ndarray,
    risk_aversion: float,
) -> float:
    if len(return_samples) == 0:
        return math.nan
    utility = exponential_utility_values(return_samples, risk_aversion)
    return float(np.mean(utility))


def exponential_utility_values(return_samples: np.ndarray, risk_aversion: float) -> np.ndarray:
    eta = float(risk_aversion)
    if eta <= 0.0:
        return np.asarray(return_samples, dtype=float)
    scaled_returns = np.clip(-eta * np.asarray(return_samples, dtype=float), -50.0, 50.0)
    return (1.0 - np.exp(scaled_returns)) / eta


def compute_cpt_objective(
    relative_returns: list[float],
    config: ExperimentConfig,
) -> float:
    if not relative_returns:
        return math.nan
    order = np.argsort(relative_returns)
    sorted_returns = [relative_returns[idx] for idx in order]
    m = len(sorted_returns)
    loss_count = sum(value < 0 for value in sorted_returns)
    objective = 0.0
    for position, rel in enumerate(sorted_returns, start=1):
        if position <= loss_count:
            delta = cpt_probability_weight(position / m, config.beta_loss) - cpt_probability_weight((position - 1) / m, config.beta_loss)
            objective += loss_value(rel, config) * delta
        else:
            upper = (m - position + 1) / m
            lower = (m - position) / m
            delta = cpt_probability_weight(upper, config.beta_gain) - cpt_probability_weight(lower, config.beta_gain)
            objective += gain_value(rel, config) * delta
    return float(objective)
