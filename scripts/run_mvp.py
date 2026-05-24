from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from dataclasses import replace

from mvp_cpt_pg.config import DateWindow, ExperimentConfig
from mvp_cpt_pg.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MVP multi-agent CPT-PG experiment")
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of methods to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Subset of seeds to run")
    parser.add_argument("--dry-run-days", type=int, default=None, help="Limit evaluation to the first N trading days")
    parser.add_argument("--llm-model", default=None, help="Override the default DashScope model")
    parser.add_argument("--llm-base-url", default=None, help="Override the OpenAI-compatible base URL")
    parser.add_argument("--disable-llm-thinking", action="store_true", help="Disable model thinking mode")
    parser.add_argument(
        "--initial-holdings",
        type=Path,
        default=None,
        help="CSV with ts_code,buy_price,shares for initial reference point and starting weights",
    )
    parser.add_argument("--initial-capital", type=float, default=None, help="Cash budget when the user has no current holdings")
    parser.add_argument("--preference-path", type=Path, default=None, help="CSV preference path generated for one synthetic user")
    parser.add_argument(
        "--disable-preference-constraints",
        action="store_true",
        help="Disable user hard-constraint projection while preserving style_tilt in policy feature scoring",
    )
    parser.add_argument(
        "--universe-by-date-path",
        type=Path,
        default=None,
        help="CSV cache with trade_date,ts_code columns for survivorship-free daily tradable universe filtering",
    )
    parser.add_argument("--evaluation-horizon", type=int, default=None, help="Sliding historical window length h for CPT-PG gradient estimation")
    parser.add_argument("--prewarm-start", default=None, help="Override prewarm start date, YYYYMMDD")
    parser.add_argument("--prewarm-end", default=None, help="Override prewarm end date, YYYYMMDD")
    parser.add_argument("--evaluation-start", default=None, help="Override evaluation start date, YYYYMMDD")
    parser.add_argument("--evaluation-end", default=None, help="Override evaluation end date, YYYYMMDD")
    parser.add_argument(
        "--strict-drop-missing-stocks",
        action="store_true",
        help="Drop stocks with any missing required non-news field after skipping empty daily_basic dates",
    )
    parser.add_argument(
        "--enable-tushare-news",
        action="store_true",
        help="Add Tushare short-news sources to the Akshare news feature pipeline",
    )
    parser.add_argument(
        "--gradient-diagnostic-repeats",
        type=int,
        default=None,
        help="Independent CPT-PG gradient estimates per update for epsilon_t diagnostics",
    )
    parser.add_argument(
        "--gradient-diagnostic-use-mean-update",
        action="store_true",
        help="Update CPT-PG theta with the mean of diagnostic gradient estimates",
    )
    parser.add_argument(
        "--cpt-sample-base",
        type=int,
        default=None,
        help="Initial n_t sample count for independent CPT objective and tail-probability samples",
    )
    parser.add_argument(
        "--gradient-sample-base",
        type=int,
        default=None,
        help="Initial m_t sample count for independent CPT-PG score-function gradient samples",
    )
    parser.add_argument(
        "--fixed-sample-counts",
        action="store_true",
        help="Keep n_t and m_t fixed at their base values for ablation experiments",
    )
    parser.add_argument(
        "--exponential-risk-aversion",
        type=float,
        default=None,
        help="Risk aversion eta for exponential_utility_pg utility (1 - exp(-eta * return)) / eta",
    )
    parser.add_argument("--eta-gain", type=float, default=None, help="Reference point gain-side adaptation rate eta_plus")
    parser.add_argument("--eta-loss", type=float, default=None, help="Reference point loss-side adaptation rate eta_minus")
    parser.add_argument("--gamma0", type=float, default=None, help="Initial policy-gradient step size")
    parser.add_argument("--gamma-exponent", type=float, default=None, help="Decay exponent for gamma_t = gamma0 / t^a")
    parser.add_argument("--policy-noise-scale", type=float, default=None, help="Gaussian policy sampling noise scale")
    parser.add_argument(
        "--policy-normalizer",
        choices=("softmax", "sparsemax"),
        default=None,
        help="Map latent policy scores to raw portfolio weights before constraint projection",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.prewarm_start or args.prewarm_end:
        config = replace(
            config,
            prewarm=DateWindow(
                start=args.prewarm_start or config.prewarm.start,
                end=args.prewarm_end or config.prewarm.end,
            ),
        )
    if args.evaluation_start or args.evaluation_end:
        config = replace(
            config,
            evaluation=DateWindow(
                start=args.evaluation_start or config.evaluation.start,
                end=args.evaluation_end or config.evaluation.end,
            ),
        )
    if args.initial_capital is not None:
        config = replace(config, initial_capital_amount=args.initial_capital)
    if args.initial_holdings is not None:
        config = replace(config, initial_holdings_path=args.initial_holdings)
    if args.preference_path is not None:
        config = replace(config, preference_path=args.preference_path)
    if args.disable_preference_constraints:
        config = replace(config, disable_preference_constraints=True)
    if args.universe_by_date_path is not None:
        config = replace(config, universe_by_date_path=args.universe_by_date_path)
    if args.evaluation_horizon is not None:
        config = replace(config, evaluation_horizon=args.evaluation_horizon)
    if args.strict_drop_missing_stocks:
        config = replace(config, strict_drop_missing_stocks=True)
    if args.enable_tushare_news:
        config = replace(config, enable_tushare_news=True)
    if args.gradient_diagnostic_repeats is not None:
        config = replace(config, gradient_diagnostic_repeats=args.gradient_diagnostic_repeats)
    if args.gradient_diagnostic_use_mean_update:
        config = replace(config, gradient_diagnostic_use_mean_update=True)
    if args.cpt_sample_base is not None:
        config = replace(config, cpt_sample_base=args.cpt_sample_base)
    if args.gradient_sample_base is not None:
        config = replace(config, gradient_sample_base=args.gradient_sample_base)
    if args.fixed_sample_counts:
        config = replace(config, fixed_sample_counts=True)
    if args.exponential_risk_aversion is not None:
        config = replace(config, exponential_risk_aversion=args.exponential_risk_aversion)
    if args.eta_gain is not None:
        config = replace(config, eta_gain=args.eta_gain)
    if args.eta_loss is not None:
        config = replace(config, eta_loss=args.eta_loss)
    if args.gamma0 is not None:
        config = replace(config, gamma0=args.gamma0)
    if args.gamma_exponent is not None:
        config = replace(config, gamma_exponent=args.gamma_exponent)
    if args.policy_noise_scale is not None:
        config = replace(config, policy_noise_scale=args.policy_noise_scale)
    if args.policy_normalizer is not None:
        config = replace(config, policy_normalizer=args.policy_normalizer)
    if args.llm_model or args.llm_base_url or args.disable_llm_thinking:
        config = _override_llm_config(
            config,
            model=args.llm_model,
            base_url=args.llm_base_url,
            enable_thinking=False if args.disable_llm_thinking else None,
        )
    runner = ExperimentRunner(config)
    artifacts = runner.run(methods=args.methods, seeds=args.seeds, dry_run_days=args.dry_run_days)
    print(f"result_dir: {artifacts.result_dir}")
    print(f"trace: {artifacts.trace_path}")
    print(f"summary_by_run: {artifacts.summary_path}")
    print(f"summary_by_method: {artifacts.aggregate_path}")
    print(f"stock_info_source_status: {artifacts.source_status_path}")
    for plot_path in artifacts.plot_paths:
        print(f"plot: {plot_path}")


def _override_llm_config(
    config: ExperimentConfig,
    *,
    model: str | None,
    base_url: str | None,
    enable_thinking: bool | None,
) -> ExperimentConfig:
    def override_agent(agent_config):
        updates = {}
        if model is not None:
            updates["model"] = model
        if base_url is not None:
            updates["base_url"] = base_url
        if enable_thinking is not None:
            updates["enable_thinking"] = enable_thinking
        return replace(agent_config, **updates)

    return replace(
        config,
        llm_model=model or config.llm_model,
        user_agent_llm=override_agent(config.user_agent_llm),
        preference_agent_llm=override_agent(config.preference_agent_llm),
        advisor_agent_llm=override_agent(config.advisor_agent_llm),
    )


if __name__ == "__main__":
    main()
