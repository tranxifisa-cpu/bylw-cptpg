from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import digamma, expit

from .actions import (
    CASH_CODE,
    ContinuousPolicyState,
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
    account_value: float
    cash_value: float
    transaction_cost: float
    traded_value: float
    objective_estimate: float
    offline_cpt_common_ref: float
    gradient_norm: float
    projected_gradient_mapping_norm: float
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
    end_account_value: float
    end_portfolio_value: float
    cash_value: float
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
        self.account_value = float(config.initial_capital_amount)
        self.portfolio_value = 0.0
        self.budget_limit = float(config.initial_capital_amount)
        self.performance_base = float(config.initial_capital_amount)
        self.reference_point = 0.0
        self.seen_dates: list[str] = []
        self.last_gradient_norm = 0.0
        self.last_projected_gradient_mapping_norm = math.nan
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
        account_value: float | None = None,
    ) -> None:
        self.reference_point = float(reference_point)
        if weights is not None:
            self.previous_weights = weights.reindex(self.asset_codes, fill_value=0.0)
        if portfolio_value is not None:
            self.portfolio_value = float(portfolio_value)
        if account_value is not None:
            self.account_value = float(account_value)
            self.performance_base = float(account_value)

    def trade_capital(self) -> float:
        return float(self.account_value)

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
        self.account_value = outcome.end_account_value
        self.portfolio_value = outcome.end_portfolio_value
        return StrategyStep(
            day_return=outcome.day_pnl,
            day_return_rate=outcome.day_return_rate,
            investment_return_rate=outcome.investment_return_rate,
            day_relative_return_rate=outcome.investment_return_rate - objective_reference,
            portfolio_value=outcome.end_portfolio_value,
            account_value=outcome.end_account_value,
            cash_value=outcome.cash_value,
            transaction_cost=outcome.transaction_cost,
            traded_value=outcome.traded_value,
            objective_estimate=self.last_objective_estimate,
            offline_cpt_common_ref=self.last_offline_cpt_common_ref,
            gradient_norm=self.last_gradient_norm,
            projected_gradient_mapping_norm=self.last_projected_gradient_mapping_norm,
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
            constraint_violation=0
            if self.config.preference_features_only
            else check_constraint_violation(weights, hard_constraints, previous_weights),
            constraint_violation_reason=""
            if self.config.preference_features_only
            else constraint_violation_reason(weights, hard_constraints, previous_weights),
        )

    def _update_model(self, trade_date: str, preference: PreferenceVector, hard_constraints: HardConstraints) -> None:
        self.last_gradient_norm = 0.0
        self.last_projected_gradient_mapping_norm = math.nan
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
        self._fixed_asset_codes: list[str] | None = None

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
        observed_state = self.dataset.observed_stock_state(trade_date)
        policy_state = build_continuous_policy_state(
            observed_state,
            pref,
            prev_weights=self.previous_weights,
            use_style_tilt=self.config.preference_features_enabled,
        )
        policy_state = self._restrict_to_fixed_asset_pool(policy_state)
        self._update_model(trade_date, pref, hard_constraints)
        weights, _, metadata = self._sample_continuous_action(policy_state, hard_constraints, self.previous_weights)
        return PortfolioDecision(
            action_name="continuous_weight_policy",
            weights=weights,
            metadata=metadata,
        )

    def _restrict_to_fixed_asset_pool(self, policy_state: ContinuousPolicyState) -> ContinuousPolicyState:
        fixed_count = self.config.fixed_asset_count
        if fixed_count is None:
            return policy_state
        fixed_count = int(fixed_count)
        if fixed_count <= 0:
            raise RuntimeError("fixed_asset_count must be positive when configured")
        if self._fixed_asset_codes is None:
            if len(policy_state.codes) < fixed_count:
                raise RuntimeError(
                    f"Fixed asset pool requested {fixed_count} stocks, but only {len(policy_state.codes)} are available"
                )
            selected = self.rng.choice(np.asarray(policy_state.codes, dtype=object), size=fixed_count, replace=False)
            self._fixed_asset_codes = sorted(str(code) for code in selected.tolist())
        fixed_set = set(self._fixed_asset_codes)
        keep_positions = [index for index, code in enumerate(policy_state.codes) if code in fixed_set]
        if not keep_positions:
            raise RuntimeError("Fixed asset pool has no tradable stocks in the current state")
        return ContinuousPolicyState(
            codes=[policy_state.codes[index] for index in keep_positions],
            feature_matrix=policy_state.feature_matrix[keep_positions],
            frame=policy_state.frame.iloc[keep_positions].reset_index(drop=True),
        )

    def _sample_continuous_action(
        self,
        policy_state,
        hard_constraints: HardConstraints,
        prev_weights: pd.Series,
        rng: np.random.Generator | None = None,
    ) -> tuple[pd.Series, np.ndarray, dict[str, Any]]:
        active_rng = rng if rng is not None else self.rng
        mean_vector = policy_state.feature_matrix @ self.theta
        selected_indices = self._sample_asset_indices(len(policy_state.codes), active_rng)
        selected_features = policy_state.feature_matrix[selected_indices]
        selected_scores = mean_vector[selected_indices]
        if self.config.policy_normalizer == "dirichlet":
            alpha, alpha_derivative = dirichlet_alpha_from_scores(selected_scores, self.config)
            raw_weight_values = dirichlet_execution_weights(alpha, self.config, active_rng)
            score_gradient = dirichlet_score_gradient(
                selected_features,
                raw_weight_values,
                alpha,
                alpha_derivative,
            )
        else:
            if self.policy_noise_scale <= 0:
                raise RuntimeError("policy_noise_scale must be positive")
            noise = active_rng.normal(loc=0.0, scale=self.policy_noise_scale, size=len(selected_indices))
            latent = selected_scores + noise
            raw_weight_values = normalize_latent_vector(latent, self.config.policy_normalizer, self.config.policy_temperature)
            latent_score = (latent - selected_scores) / (self.policy_noise_scale**2)
            score_gradient = selected_features.T @ latent_score
        raw_full_weights = aggregate_sampled_asset_weights(len(policy_state.codes), selected_indices, raw_weight_values)
        raw_risky_weights = pd.Series(raw_full_weights, index=policy_state.codes, dtype=float)
        projection_codes = [*policy_state.codes, CASH_CODE]
        raw_weights = raw_risky_weights.reindex(projection_codes, fill_value=0.0)
        if self.config.preference_features_only:
            weights = normalize_weights(raw_weights)
        else:
            weights = project_continuous_weights(
                codes=projection_codes,
                raw_weights=raw_weights,
                hard_constraints=hard_constraints,
                prev_weights=prev_weights.reindex(projection_codes, fill_value=0.0),
            )
        action_summary = continuous_action_summary(
            weights=weights,
            prev_weights=prev_weights,
            portfolio_value=self.account_value,
            display_threshold=1e-12 if self.config.preference_features_only else None,
        )
        metadata = {
            "policy_mean_vector": mean_vector,
            "action_summary": action_summary,
            "bootstrap_asset_count": int(len(selected_indices)),
        }
        return weights, score_gradient, metadata

    def _sample_asset_indices(self, asset_count: int, rng: np.random.Generator) -> np.ndarray:
        if asset_count <= 0:
            raise RuntimeError("Policy state has no risky assets")
        sample_count = self.config.bootstrap_asset_count
        if sample_count is None:
            return np.arange(asset_count, dtype=int)
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise RuntimeError("bootstrap_asset_count must be positive when configured")
        return rng.integers(0, asset_count, size=sample_count, endpoint=False, dtype=int)

    def _sample_continuous_action_batch(
        self,
        policy_state,
        hard_constraints: HardConstraints,
        prev_weight_matrix: np.ndarray,
        prev_codes: list[str],
        rng: np.random.Generator,
        collect_scores: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_vector = policy_state.feature_matrix @ self.theta
        if self.config.bootstrap_asset_count is None and self.config.policy_normalizer == "dirichlet":
            alpha, alpha_derivative = dirichlet_alpha_from_scores(mean_vector, self.config)
            raw_risky_weight_matrix = rng.dirichlet(alpha, size=prev_weight_matrix.shape[0])
        elif self.config.bootstrap_asset_count is None:
            if self.policy_noise_scale <= 0:
                raise RuntimeError("policy_noise_scale must be positive")
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
                self.config.policy_temperature,
            )
        else:
            raw_risky_weight_matrix = None
        projection_codes = [*policy_state.codes, CASH_CODE]
        aligned_prev = align_weight_matrix(prev_weight_matrix, prev_codes, projection_codes)
        sample_count = prev_weight_matrix.shape[0]
        projected_weight_matrix = np.empty((sample_count, len(projection_codes)), dtype=float)
        score_gradient_matrix = (
            np.zeros((sample_count, len(POLICY_FEATURE_COLUMNS)), dtype=float)
            if collect_scores and self.config.bootstrap_asset_count is not None
            else None
        )
        raw_full_weights = np.zeros(len(policy_state.codes), dtype=float)
        raw_weights = np.zeros(len(projection_codes), dtype=float)
        for row_index in range(sample_count):
            if self.config.bootstrap_asset_count is not None:
                selected_indices = self._sample_asset_indices(len(policy_state.codes), rng)
                selected_features = policy_state.feature_matrix[selected_indices]
                selected_scores = mean_vector[selected_indices]
                if self.config.policy_normalizer == "dirichlet":
                    alpha, alpha_derivative = dirichlet_alpha_from_scores(selected_scores, self.config)
                    raw_selected_weights = rng.dirichlet(alpha)
                    if collect_scores and score_gradient_matrix is not None:
                        score_gradient_matrix[row_index] = dirichlet_score_gradient(
                            selected_features,
                            raw_selected_weights,
                            alpha,
                            alpha_derivative,
                        )
                else:
                    if self.policy_noise_scale <= 0:
                        raise RuntimeError("policy_noise_scale must be positive")
                    noise = rng.normal(loc=0.0, scale=self.policy_noise_scale, size=len(selected_indices))
                    latent = selected_scores + noise
                    raw_selected_weights = normalize_latent_vector(
                        latent,
                        self.config.policy_normalizer,
                        self.config.policy_temperature,
                    )
                    if collect_scores and score_gradient_matrix is not None:
                        latent_score = (latent - selected_scores) / (self.policy_noise_scale**2)
                        score_gradient_matrix[row_index] = selected_features.T @ latent_score
                raw_full_weights = aggregate_sampled_asset_weights(
                    len(policy_state.codes),
                    selected_indices,
                    raw_selected_weights,
                    out=raw_full_weights,
                )
            else:
                raw_full_weights = raw_risky_weight_matrix[row_index]
            raw_weights.fill(0.0)
            raw_weights[: len(policy_state.codes)] = raw_full_weights
            if self.config.preference_features_only:
                projected_weight_matrix[row_index] = normalize_weight_array(raw_weights)
            else:
                projected = project_continuous_weights_array(
                    codes=projection_codes,
                    raw_weights=raw_weights,
                    hard_constraints=hard_constraints,
                    prev_weights=aligned_prev[row_index],
                )
                projected_weight_matrix[row_index] = projected
        if not collect_scores:
            return projected_weight_matrix, np.empty((0, len(POLICY_FEATURE_COLUMNS)), dtype=float)
        if score_gradient_matrix is not None:
            return projected_weight_matrix, score_gradient_matrix
        if self.config.policy_normalizer == "dirichlet":
            score_gradient_matrix = dirichlet_score_gradient_matrix(
                policy_state.feature_matrix,
                raw_risky_weight_matrix,
                alpha,
                alpha_derivative,
            )
        else:
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
        projected_gradient_mapping_norm = float(np.linalg.norm(theta_after - theta_before) / max(gamma_t, 1e-12))
        self.theta = theta_after
        self.last_gradient_norm = gradient_norm
        self.last_projected_gradient_mapping_norm = projected_gradient_mapping_norm
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
        cpt_relative_returns = cpt_returns - objective_reference
        gradient_relative_returns = gradient_returns - objective_reference
        common_reference_returns = cpt_returns - self.config.offline_cpt_reference
        if score_matrix is None:
            raise RuntimeError("Gradient sample scores were not collected")
        gradient_score_vectors = [score_matrix[row_index] for row_index in range(m_t)]
        gradient = compute_cpt_gradient(
            gradient_relative_returns.tolist(),
            gradient_score_vectors,
            cpt_relative_returns.tolist(),
            self.config,
        )
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
                use_style_tilt=self.config.preference_features_enabled,
            )
            policy_state = self._restrict_to_fixed_asset_pool(policy_state)
            weight_matrix, score_gradient_matrix = self._sample_continuous_action_batch(
                policy_state,
                hard_constraints,
                prev_weight_matrix,
                prev_codes,
                rng,
                collect_scores=collect_scores,
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
            turnover = np.abs(weight_matrix[:, risky_columns] - aligned_prev[:, risky_columns]).sum(axis=1)
            cash_index = projection_codes.index(CASH_CODE)
            mu = transaction_remainder_factor(
                aligned_prev,
                weight_matrix,
                self.config.trade_cost_bps / 10000.0,
                cash_index=cash_index,
            )
            investable_values = simulated_values * mu
            transaction_cost = np.maximum(simulated_values - investable_values, 0.0)
            invested_values = investable_values * risky_weight_sums
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


def dirichlet_alpha_from_scores(scores: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    alpha_min = float(config.dirichlet_alpha_min)
    alpha_max = float(config.dirichlet_alpha_max)
    if not (0.0 < alpha_min < alpha_max):
        raise RuntimeError("Dirichlet alpha bounds must satisfy 0 < alpha_min < alpha_max")
    sigmoid_values = expit(np.asarray(scores, dtype=float))
    span = alpha_max - alpha_min
    alpha = alpha_min + span * sigmoid_values
    alpha_derivative = span * sigmoid_values * (1.0 - sigmoid_values)
    return alpha.astype(float), alpha_derivative.astype(float)


def dirichlet_score_gradient(
    feature_matrix: np.ndarray,
    weights: np.ndarray,
    alpha: np.ndarray,
    alpha_derivative: np.ndarray,
) -> np.ndarray:
    safe_weights = np.maximum(np.asarray(weights, dtype=float), 1e-300)
    coefficient = (digamma(float(np.sum(alpha))) - digamma(alpha) + np.log(safe_weights)) * alpha_derivative
    return np.asarray(feature_matrix, dtype=float).T @ coefficient


def dirichlet_score_gradient_matrix(
    feature_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    alpha: np.ndarray,
    alpha_derivative: np.ndarray,
) -> np.ndarray:
    safe_weights = np.maximum(np.asarray(weight_matrix, dtype=float), 1e-300)
    coefficient_matrix = (
        digamma(float(np.sum(alpha))) - digamma(alpha) + np.log(safe_weights)
    ) * alpha_derivative.reshape(1, -1)
    return coefficient_matrix @ np.asarray(feature_matrix, dtype=float)


def dirichlet_execution_weights(alpha: np.ndarray, config: ExperimentConfig, rng: np.random.Generator) -> np.ndarray:
    mode = str(config.dirichlet_execution_mode)
    alpha_values = np.asarray(alpha, dtype=float)
    if mode == "sample":
        return rng.dirichlet(alpha_values)
    if mode == "mean":
        total = float(alpha_values.sum())
        if total <= 0.0:
            raise RuntimeError("Dirichlet alpha sum must be positive")
        return alpha_values / total
    raise ValueError(f"Unknown dirichlet_execution_mode: {mode}")


def aggregate_sampled_asset_weights(
    asset_count: int,
    selected_indices: np.ndarray,
    selected_weights: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if out is None:
        output = np.zeros(asset_count, dtype=float)
    else:
        output = out
        output.fill(0.0)
    np.add.at(output, np.asarray(selected_indices, dtype=int), np.asarray(selected_weights, dtype=float))
    return output


def softmax_vector(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp_vals = np.exp(np.clip(shifted, -30.0, 30.0))
    denom = float(exp_vals.sum())
    if denom <= 0:
        raise RuntimeError("Continuous policy softmax normalization failed")
    return exp_vals / denom


def policy_weight_matrix_from_latent(
    codes: list[str],
    latent_matrix: np.ndarray,
    normalizer: str = "softmax",
    temperature: float = 1.0,
) -> np.ndarray:
    if latent_matrix.ndim != 2 or latent_matrix.shape[1] != len(codes):
        raise RuntimeError("Policy codes and latent matrix shape mismatch")
    return normalize_latent_matrix(latent_matrix, normalizer, temperature)


def normalize_latent_vector(values: np.ndarray, normalizer: str, temperature: float = 1.0) -> np.ndarray:
    scaled = scale_latent_values(values, temperature)
    if normalizer == "softmax":
        return softmax_vector(scaled)
    if normalizer == "sparsemax":
        return sparsemax_vector(scaled)
    raise ValueError(f"Unknown policy_normalizer: {normalizer}")


def normalize_latent_matrix(values: np.ndarray, normalizer: str, temperature: float = 1.0) -> np.ndarray:
    scaled = scale_latent_values(values, temperature)
    if normalizer == "softmax":
        return softmax_matrix(scaled)
    if normalizer == "sparsemax":
        return sparsemax_matrix(scaled)
    raise ValueError(f"Unknown policy_normalizer: {normalizer}")


def scale_latent_values(values: np.ndarray, temperature: float) -> np.ndarray:
    active_temperature = float(temperature)
    if active_temperature <= 0.0:
        raise RuntimeError("policy_temperature must be positive")
    return np.asarray(values, dtype=float) / active_temperature


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


def normalize_weight_array(weights: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(clipped.sum())
    if total <= 0.0:
        raise RuntimeError("Policy weights are non-positive")
    return clipped / total


def one_way_traded_notional(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    all_codes = sorted(code for code in set(prev_weights.index).union(new_weights.index) if code != CASH_CODE)
    prev = prev_weights.reindex(all_codes, fill_value=0.0)
    new = new_weights.reindex(all_codes, fill_value=0.0)
    delta = new - prev
    buy_notional = float(delta.clip(lower=0.0).sum())
    sell_notional = float((-delta).clip(lower=0.0).sum())
    return max(buy_notional, sell_notional)


def transaction_remainder_factor(
    prev_weights: np.ndarray,
    target_weights: np.ndarray,
    commission_rate: float,
    cash_index: int = 0,
) -> np.ndarray:
    previous = np.asarray(prev_weights, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    if previous.shape != target.shape:
        raise RuntimeError("Transaction remainder factor weight shape mismatch")
    if previous.ndim == 1:
        previous = previous.reshape(1, -1)
        target = target.reshape(1, -1)
        squeeze = True
    elif previous.ndim == 2:
        squeeze = False
    else:
        raise RuntimeError("Transaction remainder factor expects 1D or 2D weights")
    rate = float(max(0.0, commission_rate))
    if rate <= 0.0:
        result = np.ones(previous.shape[0], dtype=float)
        return result[0] if squeeze else result
    cash_position = int(cash_index)
    if cash_position < 0 or cash_position >= previous.shape[1]:
        raise RuntimeError("Transaction remainder factor cash index is out of range")
    risky_positions = np.array([index for index in range(previous.shape[1]) if index != cash_position], dtype=int)
    target_cash = np.clip(target[:, cash_position], 0.0, 1.0)
    prev_risky = np.clip(previous[:, risky_positions], 0.0, 1.0)
    target_risky = np.clip(target[:, risky_positions], 0.0, 1.0)
    mu = np.ones(previous.shape[0], dtype=float)
    denominator = np.maximum(1.0 - rate * target_cash, 1e-12)
    for _ in range(32):
        sell_amount = np.maximum(prev_risky - mu.reshape(-1, 1) * target_risky, 0.0).sum(axis=1)
        next_mu = (1.0 - rate * target_cash - (2.0 * rate - rate * rate) * sell_amount) / denominator
        next_mu = np.clip(next_mu, 0.0, 1.0)
        if float(np.max(np.abs(next_mu - mu))) < 1e-12:
            mu = next_mu
            break
        mu = next_mu
    return float(mu[0]) if squeeze else mu


def portfolio_step_value(
    day_returns: pd.Series,
    prev_weights: pd.Series,
    weights: pd.Series,
    portfolio_value: float,
    trade_cost_bps: float,
) -> PortfolioOutcome:
    weights = normalize_weights(weights)
    current_value = max(float(portfolio_value), 0.0)
    traded_value = current_value * one_way_traded_notional(prev_weights, weights)
    all_codes = [CASH_CODE, *sorted(code for code in set(prev_weights.index).union(weights.index) if code != CASH_CODE)]
    prev_array = normalize_weights(prev_weights.reindex(all_codes, fill_value=0.0)).to_numpy(dtype=float)
    target_array = normalize_weights(weights.reindex(all_codes, fill_value=0.0)).to_numpy(dtype=float)
    mu = transaction_remainder_factor(prev_array, target_array, trade_cost_bps / 10000.0)
    investable_value = current_value * mu
    transaction_cost = max(current_value - investable_value, 0.0)
    stock_weights = weights.drop(labels=[CASH_CODE], errors="ignore")
    risky_weight_sum = float(stock_weights.clip(lower=0.0).sum())
    invested_value = investable_value * risky_weight_sum
    weighted_return = float((stock_weights.reindex(day_returns.index, fill_value=0.0) * day_returns.fillna(0.0)).sum())
    cash_weight = max(float(weights.get(CASH_CODE, 0.0)), 0.0)
    cash_value = investable_value * cash_weight
    end_portfolio_value = investable_value * (risky_weight_sum + weighted_return)
    end_account_value = cash_value + end_portfolio_value
    day_pnl = end_account_value - current_value
    day_return_rate = 0.0 if current_value <= 0 else day_pnl / current_value
    investment_return_rate = 0.0 if invested_value <= 1e-12 else day_pnl / invested_value
    turnover_ratio = 0.0 if end_portfolio_value <= 1e-12 else traded_value / end_portfolio_value
    return PortfolioOutcome(
        end_value=end_account_value,
        end_account_value=end_account_value,
        end_portfolio_value=end_portfolio_value,
        cash_value=cash_value,
        day_pnl=day_pnl,
        day_return_rate=day_return_rate,
        investment_return_rate=investment_return_rate,
        invested_value=invested_value,
        transaction_cost=transaction_cost,
        traded_value=traded_value,
        turnover=turnover_ratio,
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


def cpt_probability_weight_derivative(probability: float, beta: float, probability_floor: float = 1e-8) -> float:
    floor = min(0.5, max(1e-8, float(probability_floor)))
    p = min(1.0 - floor, max(floor, probability))
    b = float(beta)
    p_beta = p**b
    q_beta = (1.0 - p) ** b
    total = p_beta + q_beta
    weight = p_beta / (total ** (1.0 / b))
    log_derivative = (b / p) - ((p ** (b - 1.0)) - ((1.0 - p) ** (b - 1.0))) / total
    return float(weight * log_derivative)


def gain_utility_magnitude(relative_return: float, config: ExperimentConfig) -> float:
    return smoothed_power_value(max(relative_return, 0.0), config.alpha_gain, config.cpt_value_smoothing)


def loss_utility_magnitude(relative_return: float, config: ExperimentConfig) -> float:
    return float(config.loss_aversion * smoothed_power_value(max(-relative_return, 0.0), config.alpha_loss, config.cpt_value_smoothing))


def quantile_tail_sum(target_utility: float, sample_utilities: np.ndarray, beta: float) -> float:
    target = float(max(target_utility, 0.0))
    if target <= 0.0:
        return 0.0
    samples = np.sort(np.maximum(np.asarray(sample_utilities, dtype=float), 0.0))
    if samples.size == 0:
        return 0.0
    n = float(samples.size)
    probability_floor = 1.0 / (n + 1.0)
    lower = 0.0
    estimate = 0.0
    for index, value in enumerate(samples):
        upper = min(float(value), target)
        if upper <= lower:
            if float(value) >= target:
                break
            continue
        tail_probability = (samples.size - index) / n
        estimate += (upper - lower) * cpt_probability_weight_derivative(tail_probability, beta, probability_floor)
        lower = upper
        if lower >= target:
            break
    if lower < target:
        estimate += (target - lower) * cpt_probability_weight_derivative(0.0, beta, probability_floor)
    return float(estimate)


def cpt_policy_gradient_weight(
    relative_return: float,
    cpt_gain_utilities: np.ndarray,
    cpt_loss_utilities: np.ndarray,
    config: ExperimentConfig,
) -> float:
    gain_component = quantile_tail_sum(
        gain_utility_magnitude(relative_return, config),
        cpt_gain_utilities,
        config.beta_gain,
    )
    loss_component = quantile_tail_sum(
        loss_utility_magnitude(relative_return, config),
        cpt_loss_utilities,
        config.beta_loss,
    )
    return float(gain_component - loss_component)


def compute_cpt_gradient(
    relative_returns: list[float],
    score_vectors: list[np.ndarray],
    cpt_relative_returns: list[float],
    config: ExperimentConfig,
) -> np.ndarray:
    if not relative_returns:
        return np.zeros(len(POLICY_FEATURE_COLUMNS), dtype=float)
    if len(relative_returns) != len(score_vectors):
        raise RuntimeError("CPT gradient samples and score vectors length mismatch")
    if not cpt_relative_returns:
        raise RuntimeError("CPT quantile samples are required for CPT policy gradient estimation")
    m = len(relative_returns)
    gradient = np.zeros_like(score_vectors[0], dtype=float)
    cpt_samples = np.asarray(cpt_relative_returns, dtype=float)
    cpt_gain_utilities = np.array([gain_utility_magnitude(value, config) for value in cpt_samples], dtype=float)
    cpt_loss_utilities = np.array([loss_utility_magnitude(value, config) for value in cpt_samples], dtype=float)
    for rel, score in zip(relative_returns, score_vectors):
        psi = cpt_policy_gradient_weight(rel, cpt_gain_utilities, cpt_loss_utilities, config)
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
