from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .actions import (
    CASH_CODE,
    MIN_ACTIVE_WEIGHT,
    POLICY_FEATURE_COLUMNS,
    effective_holding_count,
    format_portfolio_weights_percent,
)
from .config import ExperimentConfig
from .market_data import MarketDataset, MarketDatasetBuilder
from .metrics import add_dynamic_local_regret_columns, aggregate_methods, summarize_runs
from .plots import generate_plots
from .progress import progress
from .strategies import build_strategy, portfolio_step_value
from .utils import ensure_dir


@dataclass
class ExperimentArtifacts:
    result_dir: Path
    trace_path: Path
    summary_path: Path
    aggregate_path: Path
    source_status_path: Path
    plot_paths: list[Path]


@dataclass
class InitialPortfolio:
    reference_point: float
    portfolio_value: float
    account_value: float
    weights: pd.Series
    price_date: str


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.dataset_builder = MarketDatasetBuilder(config)

    def run(
        self,
        methods: Iterable[str] | None = None,
        seeds: Iterable[int] | None = None,
        dry_run_days: int | None = None,
    ) -> ExperimentArtifacts:
        active_config = self.config
        active_builder = self.dataset_builder
        if dry_run_days is not None:
            eval_calendar = self.dataset_builder.tushare.trade_calendar(self.config.evaluation.start, self.config.evaluation.end)
            if not eval_calendar:
                raise RuntimeError("No evaluation trade dates were returned by Tushare")
            dry_end = eval_calendar[min(dry_run_days, len(eval_calendar)) - 1]
            active_config = replace(
                self.config,
                evaluation=replace(self.config.evaluation, end=dry_end),
            )
            active_builder = MarketDatasetBuilder(active_config)
        dataset = active_builder.build()
        methods = list(methods or self.config.methods)
        seeds = list(seeds or self.config.seeds)
        evaluation_dates = [
            trade_date
            for trade_date in dataset.trade_dates
            if active_config.evaluation.start <= trade_date <= active_config.evaluation.end
        ]
        if not evaluation_dates:
            raise RuntimeError("No evaluation dates are available")
        initial_portfolio = self._load_initial_portfolio(
            dataset,
            first_evaluation_date=evaluation_dates[0],
            path=active_config.initial_holdings_path,
        )
        result_dir = self._create_result_dir()
        trace_dir = ensure_dir(result_dir / "traces")
        table_dir = ensure_dir(result_dir / "tables")
        plot_dir = ensure_dir(result_dir / "plots")
        trace_path = trace_dir / "daily_trace.csv"
        summary_path = table_dir / "summary_by_run.csv"
        aggregate_path = table_dir / "summary_by_method.csv"
        source_status_path = table_dir / "stock_info_source_status.csv"
        plot_paths: list[Path] = []
        persisted_rows: list[dict[str, object]] = []
        self._write_config_snapshot(
            result_dir,
            active_config=active_config,
            methods=methods,
            seeds=seeds,
            evaluation_dates=evaluation_dates,
            dry_run_days=dry_run_days,
        )
        self._write_csv_atomic(dataset.source_status, source_status_path)

        def persist_row(row: dict[str, object]) -> None:
            nonlocal plot_paths
            persisted_rows.append(row)
            trace = self._add_tracking_metrics(pd.DataFrame(persisted_rows), active_config)
            summary = summarize_runs(trace)
            aggregate = aggregate_methods(summary)
            self._write_csv_atomic(trace, trace_path)
            self._write_csv_atomic(summary, summary_path)
            self._write_csv_atomic(aggregate, aggregate_path)
            self._write_csv_atomic(dataset.source_status, source_status_path)
            plot_paths = generate_plots(trace, plot_dir)

        try:
            for method in progress(methods, desc="methods", total=len(methods)):
                for seed in progress(seeds, desc=f"{method} seeds", total=len(seeds)):
                    self._run_single(
                        dataset,
                        method,
                        seed,
                        evaluation_dates,
                        initial_portfolio,
                        on_row=persist_row,
                    )
        finally:
            if persisted_rows:
                trace = self._add_tracking_metrics(pd.DataFrame(persisted_rows), active_config)
                summary = summarize_runs(trace)
                aggregate = aggregate_methods(summary)
                self._write_csv_atomic(trace, trace_path)
                self._write_csv_atomic(summary, summary_path)
                self._write_csv_atomic(aggregate, aggregate_path)
                self._write_csv_atomic(dataset.source_status, source_status_path)
                plot_paths = generate_plots(trace, plot_dir)
        return ExperimentArtifacts(
            result_dir=result_dir,
            trace_path=trace_path,
            summary_path=summary_path,
            aggregate_path=aggregate_path,
            source_status_path=source_status_path,
            plot_paths=plot_paths,
        )

    def _run_single(
        self,
        dataset: MarketDataset,
        method: str,
        seed: int,
        evaluation_dates: list[str],
        initial_portfolio: InitialPortfolio | None,
        on_row: Callable[[dict[str, object]], None] | None = None,
    ) -> pd.DataFrame:
        run_key = f"{method}_seed{seed}"
        strategy = build_strategy(method, self.config, dataset, seed)
        if initial_portfolio is not None:
            strategy.set_initial_portfolio(
                reference_point=initial_portfolio.reference_point,
                weights=initial_portfolio.weights,
                portfolio_value=initial_portfolio.portfolio_value,
                account_value=initial_portfolio.account_value,
            )
        strategy.initialize()
        initial_value = strategy.performance_base
        wealth = 1.0
        on_policy_episode = self.config.estimation_mode == "on_policy_episode"
        if self.config.estimation_mode not in {"rolling_window", "on_policy_episode"}:
            raise RuntimeError(f"Unknown estimation_mode: {self.config.estimation_mode}")
        if self.config.reference_update_frequency not in {"daily", "episode"}:
            raise RuntimeError(f"Unknown reference_update_frequency: {self.config.reference_update_frequency}")
        if self.config.reference_update_frequency == "episode" and not on_policy_episode:
            raise RuntimeError("reference_update_frequency=episode requires estimation_mode=on_policy_episode")
        episode_dates: list[str] = []
        episode_index = 0
        episode_start_value = strategy.trade_capital()
        episode_start_weights = strategy.previous_weights.copy()
        episode_start_reference = strategy.objective_reference()
        rows = []
        for trade_date in progress(evaluation_dates, desc=run_key, total=len(evaluation_dates)):
            if on_policy_episode and not episode_dates:
                episode_index += 1
                episode_start_value = strategy.trade_capital()
                episode_start_weights = strategy.previous_weights.copy()
                episode_start_reference = strategy.objective_reference()
            decision = strategy.select(trade_date)
            action_summary = decision.metadata.get("action_summary", {"name": decision.action_name})
            previous_portfolio_value = strategy.portfolio_value
            execution_capital = strategy.trade_capital()
            outcome = portfolio_step_value(
                strategy.return_matrix.loc[trade_date],
                strategy.previous_weights,
                decision.weights,
                execution_capital,
                self.config.trade_cost_bps,
            )
            previous_value = execution_capital
            wealth = outcome.end_account_value / initial_value if initial_value > 0 else 0.0
            step = strategy.after_day(
                trade_date=trade_date,
                weights=decision.weights,
                outcome=outcome,
            )
            episode_update = False
            current_episode_index = episode_index if on_policy_episode else 0
            episode_reference_drift = 0.0
            if on_policy_episode:
                episode_dates.append(trade_date)
                if len(episode_dates) >= self.config.evaluation_horizon or trade_date == evaluation_dates[-1]:
                    if self.config.reference_update_frequency == "episode":
                        episode_reference_drift = strategy.update_reference_from_signal(wealth)
                    strategy.update_from_episode(
                        episode_dates=list(episode_dates),
                        starting_value=episode_start_value,
                        starting_weights=episode_start_weights,
                        objective_reference=episode_start_reference,
                        update_key=f"{trade_date}|episode_{episode_index}",
                    )
                    episode_dates = []
                    episode_update = True
            row = {
                "run_key": run_key,
                "method": method,
                "seed": seed,
                "trade_date": trade_date,
                "estimation_mode": self.config.estimation_mode,
                "episode_index": current_episode_index,
                "episode_update": int(episode_update),
                "action_name": decision.action_name,
                "wealth": wealth,
                "day_return": step.day_return,
                "day_return_rate": step.day_return_rate,
                "investment_return_rate": step.investment_return_rate,
                "objective_signal": step.objective_signal,
                "objective_relative_value": step.objective_relative_value,
                "day_relative_return_rate": step.day_relative_return_rate,
                "portfolio_value": step.portfolio_value,
                "account_value": step.account_value,
                "cash_value": step.cash_value,
                "previous_portfolio_value": previous_portfolio_value,
                "previous_account_value": previous_value,
                "invested_value": outcome.invested_value,
                "transaction_cost": step.transaction_cost,
                "traded_value": step.traded_value,
                "reference_point": strategy.reference_point,
                "objective_estimate": strategy.last_objective_estimate,
                "offline_cpt_common_ref": strategy.last_offline_cpt_common_ref,
                "offline_cpt_reference": self.config.offline_cpt_reference,
                "gradient_norm": strategy.last_gradient_norm,
                "projected_gradient_mapping_norm": strategy.last_projected_gradient_mapping_norm,
                "gradient_bootstrap_error_norm": strategy.last_gradient_bootstrap_error_norm,
                "gradient_bootstrap_std_norm": strategy.last_gradient_bootstrap_std_norm,
                "gradient_bootstrap_se_norm": strategy.last_gradient_bootstrap_se_norm,
                "gradient_bootstrap_relative_error": strategy.last_gradient_bootstrap_relative_error,
                "objective_bootstrap_std": strategy.last_objective_bootstrap_std,
                "gradient_diagnostic_repeats": strategy.last_gradient_diagnostic_repeats,
                "cpt_sample_count": strategy.last_cpt_sample_count,
                "gradient_sample_count": strategy.last_gradient_sample_count,
                "update_norm": strategy.last_update_norm,
                "theta_norm": strategy.last_theta_norm,
                "theta_max_abs": strategy.last_theta_max_abs,
                "theta_boundary_share": strategy.last_theta_boundary_share,
                "policy_normalizer": self.config.policy_normalizer,
                "policy_temperature": self.config.policy_temperature,
                "fixed_asset_count": self.config.fixed_asset_count,
                "bootstrap_asset_count": self.config.bootstrap_asset_count,
                "dirichlet_execution_mode": self.config.dirichlet_execution_mode,
                "policy_feature_columns": json.dumps(POLICY_FEATURE_COLUMNS, ensure_ascii=False),
                "gradient_vector": json.dumps(strategy.last_gradient_vector, ensure_ascii=False),
                "theta_before_vector": json.dumps(strategy.last_theta_before_vector, ensure_ascii=False),
                "theta_after_vector": json.dumps(strategy.last_theta_after_vector, ensure_ascii=False),
                "update_vector": json.dumps(strategy.last_update_vector, ensure_ascii=False),
                "reference_drift": step.reference_drift + episode_reference_drift,
                "turnover": step.turnover,
                "holding_count": action_summary.get(
                    "holding_count",
                    effective_holding_count(
                        decision.weights,
                        threshold=MIN_ACTIVE_WEIGHT,
                    ),
                ),
                "initial_reference_point": initial_portfolio.reference_point if initial_portfolio is not None else 1.0,
                "initial_portfolio_value": initial_portfolio.portfolio_value if initial_portfolio is not None else 0.0,
                "initial_account_value": initial_portfolio.account_value if initial_portfolio is not None else self.config.initial_capital_amount,
                "cash_budget": self.config.initial_capital_amount,
                "initial_portfolio_price_date": initial_portfolio.price_date if initial_portfolio is not None else "",
                "portfolio_weights": format_portfolio_weights_percent(
                    decision.weights,
                    threshold=MIN_ACTIVE_WEIGHT,
                ),
                "trade_plan": json.dumps(action_summary.get("trade_plan", []), ensure_ascii=False),
                "buy_amount": action_summary.get("buy_amount", 0.0),
                "sell_amount": action_summary.get("sell_amount", 0.0),
                "cash_after_trade": action_summary.get("cash_after_trade", 0.0),
            }
            rows.append(row)
            if on_row is not None:
                on_row(row)
        return pd.DataFrame(rows)

    def _add_tracking_metrics(self, trace: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
        smoothing_window = (
            config.evaluation_horizon
            if config.gradient_smoothing_window is None
            else config.gradient_smoothing_window
        )
        return add_dynamic_local_regret_columns(
            trace,
            window=max(1, int(smoothing_window)),
        )

    def _write_csv_atomic(self, frame: pd.DataFrame, path: Path) -> None:
        ensure_dir(path.parent)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        frame = self._with_percent_values(frame)
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        temp_path.replace(path)

    def _with_percent_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        percent_columns = [
            "day_return_rate",
            "investment_return_rate",
            "objective_signal",
            "objective_relative_value",
            "day_relative_return_rate",
            "reference_point",
            "initial_reference_point",
            "max_drawdown",
            "turnover",
            "turnover_mean",
            "cash_after_trade",
            "theta_boundary_share",
            "theta_boundary_share_mean",
            "reference_drift",
            "reference_drift_mean",
            "reference_path_variation",
        ]
        output = frame.copy()
        for column in percent_columns:
            if column in output.columns:
                output[column] = pd.to_numeric(output[column], errors="coerce") * 100.0
        return output

    def _create_result_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for suffix in range(100):
            name = f"run_{timestamp}" if suffix == 0 else f"run_{timestamp}_{suffix:02d}"
            candidate = self.config.result_dir / name
            if not candidate.exists():
                return ensure_dir(candidate)
        raise RuntimeError(f"Unable to create unique result directory under {self.config.result_dir}")

    def _write_config_snapshot(
        self,
        result_dir: Path,
        *,
        active_config: ExperimentConfig,
        methods: list[str],
        seeds: list[int],
        evaluation_dates: list[str],
        dry_run_days: int | None,
    ) -> None:
        snapshot = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "effective_methods": methods,
            "effective_seeds": seeds,
            "dry_run_days": dry_run_days,
            "evaluation_trade_date_count": len(evaluation_dates),
            "first_evaluation_trade_date": evaluation_dates[0] if evaluation_dates else None,
            "last_evaluation_trade_date": evaluation_dates[-1] if evaluation_dates else None,
            "config": self._json_ready(asdict(active_config)),
        }
        snapshot_path = result_dir / "config_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _json_ready(self, value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [self._json_ready(item) for item in value]
        if isinstance(value, list):
            return [self._json_ready(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._json_ready(item) for key, item in value.items()}
        return value

    def _load_initial_portfolio(
        self,
        dataset: MarketDataset,
        first_evaluation_date: str,
        path: Path | None,
    ) -> InitialPortfolio | None:
        if path is None:
            return None
        holdings_path = Path(path)
        if not holdings_path.exists():
            raise FileNotFoundError(f"Initial holdings file does not exist: {holdings_path}")
        holdings = pd.read_csv(holdings_path)
        required_columns = {"ts_code", "buy_price", "shares"}
        missing_columns = required_columns.difference(holdings.columns)
        if missing_columns:
            raise ValueError(f"Initial holdings file is missing columns: {sorted(missing_columns)}")
        holdings = holdings.copy()
        holdings["ts_code"] = holdings["ts_code"].astype(str).str.strip()
        holdings["buy_price"] = pd.to_numeric(holdings["buy_price"], errors="coerce")
        holdings["shares"] = pd.to_numeric(holdings["shares"], errors="coerce")
        invalid = holdings[holdings["ts_code"].eq("") | holdings["buy_price"].isna() | holdings["shares"].isna()]
        if not invalid.empty:
            raise ValueError("Initial holdings contain empty codes or non-numeric buy_price/shares")
        if (holdings["buy_price"] <= 0).any() or (holdings["shares"] <= 0).any():
            raise ValueError("Initial holdings buy_price and shares must be positive")
        previous_dates = [date for date in dataset.trade_dates if date < first_evaluation_date]
        if not previous_dates:
            raise RuntimeError("Cannot price initial holdings because no prior trade date is available")
        price_date = previous_dates[-1]
        price_frame = dataset.panel[dataset.panel["trade_date"] == price_date][["ts_code", "close"]].copy()
        price_frame["close"] = pd.to_numeric(price_frame["close"], errors="coerce")
        price_map = price_frame.set_index("ts_code")["close"]
        missing_codes = sorted(code for code in holdings["ts_code"] if code not in price_map.index or pd.isna(price_map.loc[code]))
        if missing_codes:
            raise ValueError(f"Initial holdings are not covered by the selected universe on {price_date}: {missing_codes}")
        holdings["current_price"] = holdings["ts_code"].map(price_map)
        holdings["cost_value"] = holdings["buy_price"] * holdings["shares"]
        holdings["market_value"] = holdings["current_price"] * holdings["shares"]
        total_cost = float(holdings["cost_value"].sum())
        total_value = float(holdings["market_value"].sum())
        if total_cost <= 0 or total_value <= 0:
            raise ValueError("Initial holdings total cost and market value must be positive")
        if total_cost > self.config.initial_capital_amount + 1e-8:
            raise ValueError(
                f"Initial holdings total cost {total_cost:.2f} exceeds budget limit {self.config.initial_capital_amount:.2f}"
            )
        remaining_cash = float(self.config.initial_capital_amount - total_cost)
        total_account_value = total_value + remaining_cash
        if total_account_value <= 0:
            raise ValueError("Initial total account value must be positive")
        weights = holdings.groupby("ts_code")["market_value"].sum()
        weights.loc[CASH_CODE] = remaining_cash
        weights = weights / float(weights.sum())
        return InitialPortfolio(
            reference_point=1.0,
            portfolio_value=total_value,
            account_value=total_account_value,
            weights=weights,
            price_date=price_date,
        )

