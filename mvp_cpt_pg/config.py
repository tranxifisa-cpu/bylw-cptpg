from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "artifacts"
CACHE_DIR = ARTIFACT_DIR / "cache"
RAW_CACHE_DIR = CACHE_DIR / "raw"
LLM_CACHE_DIR = CACHE_DIR / "llm"
RESULT_DIR = ARTIFACT_DIR / "results"
TRACE_DIR = RESULT_DIR / "traces"
PLOT_DIR = RESULT_DIR / "plots"
TABLE_DIR = RESULT_DIR / "tables"


@dataclass(frozen=True)
class DateWindow:
    start: str
    end: str


@dataclass(frozen=True)
class AgentLLMConfig:
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_envs: tuple[str, ...] = ("OPENAI_API_KEY",)
    temperature: float = 0.0
    enable_thinking: bool = True
    reasoning_effort: str | None = "max"
    thinking_budget: int = 4096
    timeout_seconds: int = 180


@dataclass(frozen=True)
class ExperimentConfig:
    prewarm: DateWindow = DateWindow(start="20250414", end="20250514")
    evaluation: DateWindow = DateWindow(start="20250515", end="20260512")
    trade_cost_bps: float = 10.0
    initial_capital_amount: float = 1_000_000.0
    seeds: tuple[int, ...] = (7, 17, 29)
    universe_size: int | None = None
    universe_industry_cap: int | None = None
    # Sliding historical window length h for CPT-PG gradient estimation.
    evaluation_horizon: int = 15
    initial_holdings_path: Path | None = None
    preference_path: Path | None = None
    strict_drop_missing_stocks: bool = False
    cpt_sample_base: int = 256
    gradient_sample_base: int = 32
    sample_exponent: float = 0.5
    fixed_sample_counts: bool = False
    gradient_diagnostic_repeats: int = 1
    gradient_diagnostic_use_mean_update: bool = False
    gamma0: float = 3.0
    gamma_exponent: float = 0.51
    policy_noise_scale: float = 0.1
    exponential_risk_aversion: float = 0.5
    alpha_gain: float = 0.88
    alpha_loss: float = 0.88
    beta_gain: float = 0.61
    beta_loss: float = 0.69
    loss_aversion: float = 2.25
    cpt_value_smoothing: float = 1.0
    eta_gain: float = 0.40
    eta_loss: float = 0.10
    offline_cpt_reference: float = 0.0
    max_logit_abs: float = 5.0
    llm_model: str = "deepseek-v4-flash"
    llm_temperature: float = 0.0
    user_agent_llm: AgentLLMConfig = field(default_factory=AgentLLMConfig)
    preference_agent_llm: AgentLLMConfig = field(default_factory=AgentLLMConfig)
    advisor_agent_llm: AgentLLMConfig = field(default_factory=AgentLLMConfig)
    news_sources: tuple[str, ...] = (
        "stock_info_global_cls",
        "stock_info_global_em",
        "stock_info_global_futu",
        "stock_info_global_sina",
        "stock_info_global_ths",
        "stock_info_cjzc_em",
    )
    enable_tushare_news: bool = False
    tushare_news_sources: tuple[str, ...] = ("sina", "10jqka", "eastmoney", "cls")
    tushare_news_chunk_days: int = 7
    static_stock_info_sources: tuple[str, ...] = (
        "stock_info_a_code_name",
        "stock_info_sh_name_code",
        "stock_info_sz_name_code",
        "stock_info_change_name",
        "stock_info_sz_change_name",
        "stock_info_sh_delist",
        "stock_info_sz_delist",
    )
    finance_industries: tuple[str, ...] = ("银行", "证券", "保险", "多元金融")
    methods: tuple[str, ...] = (
        "dynamic_cpt_pg",
        "static_cpt_pg",
        "dynamic_cpt_pg_frozen_pref",
        "static_ref_dynamic_pref_cpt_pg",
        "expected_return_pg",
        "exponential_utility_pg",
    )
    artifact_dir: Path = field(default=ARTIFACT_DIR)
    result_dir: Path = field(default=RESULT_DIR)
    raw_cache_dir: Path = field(default=RAW_CACHE_DIR)
    llm_cache_dir: Path = field(default=LLM_CACHE_DIR)
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
            evaluation_horizon=self.evaluation_horizon,
            initial_holdings_path=self.initial_holdings_path,
            preference_path=self.preference_path,
            strict_drop_missing_stocks=self.strict_drop_missing_stocks,
            cpt_sample_base=self.cpt_sample_base,
            gradient_sample_base=self.gradient_sample_base,
            sample_exponent=self.sample_exponent,
            fixed_sample_counts=self.fixed_sample_counts,
            gradient_diagnostic_repeats=self.gradient_diagnostic_repeats,
            gradient_diagnostic_use_mean_update=self.gradient_diagnostic_use_mean_update,
            gamma0=self.gamma0,
            gamma_exponent=self.gamma_exponent,
            policy_noise_scale=self.policy_noise_scale,
            exponential_risk_aversion=self.exponential_risk_aversion,
            alpha_gain=self.alpha_gain,
            alpha_loss=self.alpha_loss,
            beta_gain=self.beta_gain,
            beta_loss=self.beta_loss,
            loss_aversion=self.loss_aversion,
            cpt_value_smoothing=self.cpt_value_smoothing,
            eta_gain=self.eta_gain,
            eta_loss=self.eta_loss,
            offline_cpt_reference=self.offline_cpt_reference,
            max_logit_abs=self.max_logit_abs,
            llm_model=self.llm_model,
            llm_temperature=self.llm_temperature,
            user_agent_llm=self.user_agent_llm,
            preference_agent_llm=self.preference_agent_llm,
            advisor_agent_llm=self.advisor_agent_llm,
            news_sources=self.news_sources,
            enable_tushare_news=self.enable_tushare_news,
            tushare_news_sources=self.tushare_news_sources,
            tushare_news_chunk_days=self.tushare_news_chunk_days,
            static_stock_info_sources=self.static_stock_info_sources,
            finance_industries=self.finance_industries,
            methods=self.methods,
            artifact_dir=self.artifact_dir,
            result_dir=self.result_dir,
            raw_cache_dir=self.raw_cache_dir,
            llm_cache_dir=self.llm_cache_dir,
            trace_dir=self.trace_dir,
            plot_dir=self.plot_dir,
            table_dir=self.table_dir,
        )
