from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "artifacts"
CACHE_DIR = ARTIFACT_DIR / "cache"
RAW_CACHE_DIR = CACHE_DIR / "raw"
RESULT_DIR = ARTIFACT_DIR / "results"
TRACE_DIR = RESULT_DIR / "traces"
PLOT_DIR = RESULT_DIR / "plots"
TABLE_DIR = RESULT_DIR / "tables"


@dataclass(frozen=True)
class DateWindow:
    start: str
    end: str


@dataclass(frozen=True)
class ExperimentConfig:
    prewarm: DateWindow = DateWindow(start="20250414", end="20250514")
    evaluation: DateWindow = DateWindow(start="20250515", end="20260512")
    trade_cost_bps: float = 10.0
    initial_capital_amount: float = 1_000_000.0
    seeds: tuple[int, ...] = (7, 17, 29)
    universe_size: int | None = None
    universe_industry_cap: int | None = None
    universe_by_date_path: Path | None = None
    index_universe_code: str | None = None
    # Sliding historical window length h for CPT-PG gradient estimation.
    evaluation_horizon: int = 15
    initial_holdings_path: Path | None = None
    strict_drop_missing_stocks: bool = False
    cpt_sample_base: int = 256
    gradient_sample_base: int = 32
    shared_cpt_gradient_samples: bool = False
    sample_exponent: float = 0.5
    fixed_sample_counts: bool = False
    gradient_diagnostic_repeats: int = 1
    gradient_diagnostic_use_mean_update: bool = False
    estimation_mode: str = "rolling_window"
    gamma0: float = 3.0
    gamma_exponent: float = 0.51
    normalize_gradient_update: bool = True
    gradient_smoothing_window: int | None = None
    policy_noise_scale: float = 0.1
    policy_temperature: float = 1.0
    policy_normalizer: str = "dirichlet"
    fixed_asset_count: int | None = None
    bootstrap_asset_count: int | None = None
    dirichlet_execution_mode: str = "sample"
    exponential_risk_aversion: float = 0.5
    alpha_gain: float = 0.88
    alpha_loss: float = 0.88
    beta_gain: float = 0.61
    beta_loss: float = 0.69
    loss_aversion: float = 2.25
    cpt_value_smoothing: float = 1.0
    eta_gain: float = 0.40
    eta_loss: float = 0.10
    reference_update_frequency: str = "daily"
    offline_cpt_reference: float = 1.0
    finance_industries: tuple[str, ...] = ("银行", "证券", "保险", "多元金融")
    methods: tuple[str, ...] = (
        "dynamic_cpt_pg",
        "symmetric_cpt_pg",
        "static_cpt_pg",
        "expected_return_pg",
        "exponential_utility_pg",
    )
    artifact_dir: Path = field(default=ARTIFACT_DIR)
    result_dir: Path = field(default=RESULT_DIR)
    raw_cache_dir: Path = field(default=RAW_CACHE_DIR)
    trace_dir: Path = field(default=TRACE_DIR)
    plot_dir: Path = field(default=PLOT_DIR)
    table_dir: Path = field(default=TABLE_DIR)

    def with_single_seed(self, seed: int) -> "ExperimentConfig":
        return ExperimentConfig(
            prewarm=self.prewarm,
            evaluation=self.evaluation,
            trade_cost_bps=self.trade_cost_bps,
            initial_capital_amount=self.initial_capital_amount,
            seeds=(seed,),
            universe_size=self.universe_size,
            universe_industry_cap=self.universe_industry_cap,
            universe_by_date_path=self.universe_by_date_path,
            index_universe_code=self.index_universe_code,
            evaluation_horizon=self.evaluation_horizon,
            initial_holdings_path=self.initial_holdings_path,
            strict_drop_missing_stocks=self.strict_drop_missing_stocks,
            cpt_sample_base=self.cpt_sample_base,
            gradient_sample_base=self.gradient_sample_base,
            shared_cpt_gradient_samples=self.shared_cpt_gradient_samples,
            sample_exponent=self.sample_exponent,
            fixed_sample_counts=self.fixed_sample_counts,
            gradient_diagnostic_repeats=self.gradient_diagnostic_repeats,
            gradient_diagnostic_use_mean_update=self.gradient_diagnostic_use_mean_update,
            estimation_mode=self.estimation_mode,
            gamma0=self.gamma0,
            gamma_exponent=self.gamma_exponent,
            normalize_gradient_update=self.normalize_gradient_update,
            gradient_smoothing_window=self.gradient_smoothing_window,
            policy_noise_scale=self.policy_noise_scale,
            policy_temperature=self.policy_temperature,
            policy_normalizer=self.policy_normalizer,
            fixed_asset_count=self.fixed_asset_count,
            bootstrap_asset_count=self.bootstrap_asset_count,
            dirichlet_execution_mode=self.dirichlet_execution_mode,
            exponential_risk_aversion=self.exponential_risk_aversion,
            alpha_gain=self.alpha_gain,
            alpha_loss=self.alpha_loss,
            beta_gain=self.beta_gain,
            beta_loss=self.beta_loss,
            loss_aversion=self.loss_aversion,
            cpt_value_smoothing=self.cpt_value_smoothing,
            eta_gain=self.eta_gain,
            eta_loss=self.eta_loss,
            reference_update_frequency=self.reference_update_frequency,
            offline_cpt_reference=self.offline_cpt_reference,
            finance_industries=self.finance_industries,
            methods=self.methods,
            artifact_dir=self.artifact_dir,
            result_dir=self.result_dir,
            raw_cache_dir=self.raw_cache_dir,
            trace_dir=self.trace_dir,
            plot_dir=self.plot_dir,
            table_dir=self.table_dir,
        )
