from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .agents import AgentResult, MultiAgentSystem
from .actions import CASH_CODE, POLICY_FEATURE_COLUMNS, build_news_context, effective_holding_count, format_portfolio_weights_percent
from .config import ExperimentConfig
from .market_data import MarketDataset, MarketDatasetBuilder
from .metrics import aggregate_methods, summarize_runs
from .plots import generate_plots
from .progress import progress
from .schemas import HardConstraints, PreferenceAgentResponse
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
    weights: pd.Series
    price_date: str


class PreferencePathProvider:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Preference path file does not exist: {path}")
        frame = pd.read_csv(path, dtype={"trade_date": str})
        required_columns = {
            "trade_date",
            "risk_budget",
            "max_single_weight",
            "turnover_cap",
            "diversification_target",
            "style_tilt",
        }
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Preference path is missing columns: {sorted(missing_columns)}")
        self.path = path
        self.by_date = {
            str(row["trade_date"]): PreferenceAgentResponse.from_dict({"preference_vector": row})
            for row in frame.to_dict("records")
        }

    def get(self, trade_date: str) -> AgentResult:
        if trade_date not in self.by_date:
            raise ValueError(f"Preference path {self.path} has no row for trade_date={trade_date}")
        return AgentResult(payload=self.by_date[trade_date])


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.dataset_builder = MarketDatasetBuilder(config)
        self.agents = MultiAgentSystem(config)

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
        preference_provider = PreferencePathProvider(active_config.preference_path) if active_config.preference_path is not None else None
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
            trace = pd.DataFrame(persisted_rows)
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
                        preference_provider,
                        on_row=persist_row,
                    )
        finally:
            if persisted_rows:
                trace = pd.DataFrame(persisted_rows)
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
        preference_provider: PreferencePathProvider | None = None,
        on_row: Callable[[dict[str, object]], None] | None = None,
    ) -> pd.DataFrame:
        run_key = f"{method}_seed{seed}"
        strategy = build_strategy(method, self.config, dataset, seed)
        if initial_portfolio is not None:
            strategy.set_initial_portfolio(
                reference_point=initial_portfolio.reference_point,
                weights=initial_portfolio.weights,
                portfolio_value=initial_portfolio.portfolio_value,
            )
        strategy.initialize()
        initial_value = strategy.performance_base
        wealth = 1.0
        frozen_preference = None
        current_preference = None
        rows = []
        first_trade_date = evaluation_dates[0]
        preference_init_run_key = (
            f"dynamic_cpt_pg_seed{seed}"
            if strategy.use_cpt_optimizer and method != "dynamic_cpt_pg"
            else run_key
        )
        initial_state = dataset.observed_stock_state(first_trade_date)
        initial_news_state = dataset.observed_news_state(first_trade_date)
        initial_market_summary = self._market_summary(initial_state, initial_news_state)
        if preference_provider is None:
            initial_user_input = self.agents.simulate_user(
                run_key=preference_init_run_key,
                trade_date=first_trade_date,
                market_summary=initial_market_summary,
                reference_point=strategy.reference_point,
                budget_limit=self.config.initial_capital_amount,
                current_portfolio_value=strategy.portfolio_value,
            )
            initial_preference = self.agents.infer_preference(
                run_key=preference_init_run_key,
                trade_date=first_trade_date,
                user_response=initial_user_input.payload.as_dict(),
                previous_preference=None,
            )
        else:
            initial_preference = preference_provider.get(first_trade_date)
        if strategy.frozen_preference:
            frozen_preference = initial_preference.payload
            current_preference = frozen_preference
        else:
            current_preference = initial_preference.payload
        for trade_date in progress(evaluation_dates, desc=run_key, total=len(evaluation_dates)):
            observed_state = dataset.observed_stock_state(trade_date)
            observed_news_state = dataset.observed_news_state(trade_date)
            news_context = build_news_context(observed_state)
            market_summary = self._market_summary(observed_state, observed_news_state)
            if current_preference is None:
                raise RuntimeError("Current preference is not initialized")
            if preference_provider is not None and not strategy.frozen_preference:
                current_preference = preference_provider.get(trade_date).payload
            used_preference = current_preference.preference_vector
            used_constraints = (
                self._unrestricted_constraints(dataset)
                if self.config.disable_preference_constraints
                else current_preference.hard_constraints
            )
            decision = strategy.select(trade_date, used_preference, used_constraints)
            action_summary = decision.metadata.get("action_summary", {"name": decision.action_name})
            if preference_provider is None:
                advisor_result = self.agents.advise(
                    run_key=run_key,
                    trade_date=trade_date,
                    preference=used_preference,
                    hard_constraints=used_constraints,
                    action_name=decision.action_name,
                    action_summary=action_summary,
                    reference_point=strategy.reference_point,
                )
                recommended_action = advisor_result.payload.recommended_action
                advisor_rationale = advisor_result.payload.rationale
                preference_alignment = advisor_result.payload.preference_alignment
                risk_note = advisor_result.payload.risk_note
            else:
                recommended_action = decision.action_name
                advisor_rationale = f"preference_path={preference_provider.path.name}"
                preference_alignment = "external synthetic preference path"
                risk_note = "LLM dialogue skipped for deterministic preference-path experiment"
            execution_capital = strategy.trade_capital()
            outcome = portfolio_step_value(
                strategy.return_matrix.loc[trade_date],
                strategy.previous_weights,
                decision.weights,
                execution_capital,
                self.config.trade_cost_bps,
            )
            previous_value = execution_capital
            wealth = outcome.end_value / initial_value if initial_value > 0 else 0.0
            step = strategy.after_day(
                trade_date=trade_date,
                weights=decision.weights,
                outcome=outcome,
                preference=used_preference,
                hard_constraints=used_constraints,
            )
            if preference_provider is None:
                user_result = self.agents.simulate_feedback(
                    run_key=run_key,
                    trade_date=trade_date,
                    market_summary=market_summary,
                    reference_point=strategy.reference_point,
                    day_return=step.day_return,
                    portfolio_value=step.portfolio_value,
                    advisor_response=advisor_result.payload,
                    action_summary=action_summary,
                )
                next_preference = self.agents.infer_preference(
                    run_key=run_key,
                    trade_date=trade_date,
                    user_response=user_result.payload.as_dict(),
                    previous_preference=current_preference,
                )
                adoption = user_result.payload.adoption
                rating = user_result.payload.rating
                utterance = user_result.payload.utterance
                next_focus = user_result.payload.next_focus
            else:
                next_preference = None
                adoption = "preference_path"
                rating = float("nan")
                utterance = ""
                next_focus = used_preference.style_tilt
            row = {
                "run_key": run_key,
                "method": method,
                "seed": seed,
                "trade_date": trade_date,
                "action_name": decision.action_name,
                "wealth": wealth,
                "day_return": step.day_return,
                "day_return_rate": step.day_return_rate,
                "investment_return_rate": step.investment_return_rate,
                "day_relative_return_rate": step.day_relative_return_rate,
                "portfolio_value": step.portfolio_value,
                "previous_portfolio_value": previous_value,
                "invested_value": outcome.invested_value,
                "transaction_cost": step.transaction_cost,
                "traded_value": step.traded_value,
                "reference_point": strategy.reference_point,
                "objective_estimate": step.objective_estimate,
                "offline_cpt_common_ref": step.offline_cpt_common_ref,
                "offline_cpt_reference": self.config.offline_cpt_reference,
                "gradient_norm": step.gradient_norm,
                "gradient_bootstrap_error_norm": step.gradient_bootstrap_error_norm,
                "gradient_bootstrap_std_norm": step.gradient_bootstrap_std_norm,
                "gradient_bootstrap_se_norm": step.gradient_bootstrap_se_norm,
                "gradient_bootstrap_relative_error": step.gradient_bootstrap_relative_error,
                "objective_bootstrap_std": step.objective_bootstrap_std,
                "gradient_diagnostic_repeats": step.gradient_diagnostic_repeats,
                "cpt_sample_count": step.cpt_sample_count,
                "gradient_sample_count": step.gradient_sample_count,
                "update_norm": step.update_norm,
                "theta_norm": step.theta_norm,
                "theta_max_abs": step.theta_max_abs,
                "theta_boundary_share": step.theta_boundary_share,
                "policy_normalizer": self.config.policy_normalizer,
                "policy_feature_columns": json.dumps(POLICY_FEATURE_COLUMNS, ensure_ascii=False),
                "gradient_vector": json.dumps(step.gradient_vector, ensure_ascii=False),
                "theta_before_vector": json.dumps(step.theta_before_vector, ensure_ascii=False),
                "theta_after_vector": json.dumps(step.theta_after_vector, ensure_ascii=False),
                "update_vector": json.dumps(step.update_vector, ensure_ascii=False),
                "reference_drift": step.reference_drift,
                "turnover": step.turnover,
                "constraint_violation": step.constraint_violation,
                "constraint_violation_reason": step.constraint_violation_reason,
                "adoption": adoption,
                "rating": rating,
                "utterance": utterance,
                "next_focus": next_focus,
                "recommended_action": recommended_action,
                "advisor_rationale": advisor_rationale,
                "preference_alignment": preference_alignment,
                "risk_note": risk_note,
                "risk_budget": used_preference.risk_budget,
                "max_single_weight": used_preference.max_single_weight,
                "turnover_cap": used_preference.turnover_cap,
                "diversification_target": used_preference.diversification_target,
                "style_tilt": used_preference.style_tilt,
                "preference_constraints_disabled": int(self.config.disable_preference_constraints),
                "effective_risk_budget": used_constraints.risk_budget,
                "effective_max_single_weight": used_constraints.max_single_weight,
                "effective_turnover_cap": used_constraints.turnover_cap,
                "effective_diversification_target": used_constraints.diversification_target,
                "news_total_count": observed_news_state.get("news_total_count", 0),
                "news_total_positive": observed_news_state.get("news_total_positive", 0),
                "news_total_negative": observed_news_state.get("news_total_negative", 0),
                "stock_info_missing_sources": observed_news_state.get("stock_info_missing_sources", 0),
                "news_sentiment_score": news_context["news_sentiment_score"],
                "news_risk_pressure": news_context["news_risk_pressure"],
                "news_attention": news_context["news_attention"],
                "news_action_adjustment": action_summary.get("news_adjustment", 0.0),
                "holding_count": action_summary.get("holding_count", effective_holding_count(decision.weights)),
                "initial_reference_point": initial_portfolio.reference_point if initial_portfolio is not None else 0.0,
                "initial_portfolio_value": initial_portfolio.portfolio_value if initial_portfolio is not None else self.config.initial_capital_amount,
                "cash_budget": self.config.initial_capital_amount,
                "initial_portfolio_price_date": initial_portfolio.price_date if initial_portfolio is not None else "",
                "portfolio_weights": format_portfolio_weights_percent(decision.weights),
                "trade_plan": json.dumps(action_summary.get("trade_plan", []), ensure_ascii=False),
                "buy_amount": action_summary.get("buy_amount", 0.0),
                "sell_amount": action_summary.get("sell_amount", 0.0),
                "cash_after_trade": action_summary.get("cash_after_trade", 0.0),
            }
            rows.append(row)
            if on_row is not None:
                on_row(row)
            if strategy.frozen_preference:
                current_preference = frozen_preference
            elif next_preference is not None:
                current_preference = next_preference.payload
        return pd.DataFrame(rows)

    def _unrestricted_constraints(self, dataset: MarketDataset) -> HardConstraints:
        return HardConstraints(
            risk_budget=1.0,
            max_single_weight=1.0,
            turnover_cap=2.0,
            diversification_target=max(1, int(dataset.universe["ts_code"].nunique())),
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
            "day_relative_return_rate",
            "reference_point",
            "initial_reference_point",
            "max_drawdown",
            "adoption_rate",
            "constraint_violation_rate",
            "turnover",
            "turnover_mean",
            "risk_budget",
            "max_single_weight",
            "turnover_cap",
            "effective_risk_budget",
            "effective_max_single_weight",
            "effective_turnover_cap",
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
            reference_point=0.0,
            portfolio_value=total_account_value,
            weights=weights,
            price_date=price_date,
        )

    def _market_summary(self, state: pd.DataFrame, news_state: dict[str, object]) -> dict[str, object]:
        return {
            "mean_ret_1d": round(float(state["ret_1d"].mean()), 6),
            "mean_ret_5d": round(float(state["ret_5d"].mean()), 6),
            "mean_vol_20d": round(float(state["vol_20d"].mean()), 6),
            "turnover_rate": round(float(state["turnover_rate"].mean()), 6),
            "news_total_count": int(news_state.get("news_total_count", 0)),
            "news_total_positive": int(news_state.get("news_total_positive", 0)),
            "news_total_negative": int(news_state.get("news_total_negative", 0)),
        }
