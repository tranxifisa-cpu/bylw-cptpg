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
    parser = argparse.ArgumentParser(description="Run the CPT-PG portfolio experiment")
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of methods to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Subset of seeds to run")
    parser.add_argument("--dry-run-days", type=int, default=None, help="Limit evaluation to the first N trading days")
    parser.add_argument(
        "--initial-holdings",
        type=Path,
        default=None,
        help="CSV with ts_code,buy_price,shares for initial reference point and starting weights",
    )
    parser.add_argument("--initial-capital", type=float, default=None, help="Cash budget when the user has no current holdings")
    parser.add_argument(
        "--universe-by-date-path",
        type=Path,
        default=None,
        help="CSV cache with trade_date,ts_code columns for survivorship-free daily tradable universe filtering",
    )
    parser.add_argument(
        "--index-universe-code",
        default=None,
        help="Use fixed index constituents as the stock universe, for example 000300.SH for CSI 300",
    )
    parser.add_argument("--evaluation-horizon", type=int, default=None, help="Sliding historical window length h for CPT-PG gradient estimation")
    parser.add_argument("--prewarm-start", default=None, help="Override prewarm start date, YYYYMMDD")
    parser.add_argument("--prewarm-end", default=None, help="Override prewarm end date, YYYYMMDD")
    parser.add_argument("--evaluation-start", default=None, help="Override evaluation start date, YYYYMMDD")
    parser.add_argument("--evaluation-end", default=None, help="Override evaluation end date, YYYYMMDD")
    parser.add_argument(
        "--strict-drop-missing-stocks",
        action="store_true",
        help="Drop stocks with any missing required market field after skipping empty daily_basic dates",
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
        "--shared-cpt-gradient-samples",
        action="store_true",
        help="Use the same sampled trajectories for CPT quantiles/objective and score-function gradient estimation",
    )
    parser.add_argument(
        "--estimation-mode",
        choices=("rolling_window", "on_policy_episode"),
        default=None,
        help="CPT-PG estimator mode. rolling_window is the original historical-window mode; on_policy_episode updates after each h-day live episode",
    )
    parser.add_argument(
        "--exponential-risk-aversion",
        type=float,
        default=None,
        help="Risk aversion eta for exponential_utility_pg utility (1 - exp(-eta * return)) / eta",
    )
    parser.add_argument("--eta-gain", type=float, default=None, help="Reference point gain-side adaptation rate eta_plus")
    parser.add_argument("--eta-loss", type=float, default=None, help="Reference point loss-side adaptation rate eta_minus")
    parser.add_argument(
        "--reference-update-frequency",
        choices=("daily", "episode"),
        default=None,
        help="Reference point update frequency. episode updates once after each on-policy h-day period",
    )
    parser.add_argument("--gamma0", type=float, default=None, help="Initial policy-gradient step size")
    parser.add_argument("--gamma-exponent", type=float, default=None, help="Decay exponent for gamma_t = gamma0 / t^a")
    parser.add_argument(
        "--disable-gradient-normalization",
        action="store_true",
        help="Use the time-smoothed gradient itself for theta updates instead of its normalized direction",
    )
    parser.add_argument(
        "--gradient-smoothing-window",
        type=int,
        default=None,
        help="Window width for smoothed gradient updates and dynamic local regret; defaults to evaluation_horizon",
    )
    parser.add_argument("--policy-noise-scale", type=float, default=None, help="Gaussian policy sampling noise scale")
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=None,
        help="Temperature for softmax/sparsemax action normalization; values above 1 flatten latent score gaps",
    )
    parser.add_argument(
        "--policy-normalizer",
        choices=("dirichlet", "softmax", "sparsemax"),
        default=None,
        help="Policy action sampler. Dirichlet is the default CPT-PG portfolio policy",
    )
    parser.add_argument(
        "--bootstrap-asset-count",
        type=int,
        default=None,
        help="Randomly sample this many stock positions with replacement before policy weight sampling",
    )
    parser.add_argument(
        "--fixed-asset-count",
        type=int,
        default=None,
        help="Sample one fixed stock pool of this size at the start of each method/seed run",
    )
    parser.add_argument(
        "--dirichlet-execution-mode",
        choices=("sample", "mean"),
        default=None,
        help="How to turn Dirichlet alpha into the current recommendation after theta update",
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
    if args.universe_by_date_path is not None:
        config = replace(config, universe_by_date_path=args.universe_by_date_path)
    if args.index_universe_code is not None:
        config = replace(config, index_universe_code=args.index_universe_code)
    if args.evaluation_horizon is not None:
        config = replace(config, evaluation_horizon=args.evaluation_horizon)
    if args.strict_drop_missing_stocks:
        config = replace(config, strict_drop_missing_stocks=True)
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
    if args.shared_cpt_gradient_samples:
        config = replace(config, shared_cpt_gradient_samples=True)
    if args.estimation_mode is not None:
        config = replace(config, estimation_mode=args.estimation_mode)
    if args.exponential_risk_aversion is not None:
        config = replace(config, exponential_risk_aversion=args.exponential_risk_aversion)
    if args.eta_gain is not None:
        config = replace(config, eta_gain=args.eta_gain)
    if args.eta_loss is not None:
        config = replace(config, eta_loss=args.eta_loss)
    if args.reference_update_frequency is not None:
        config = replace(config, reference_update_frequency=args.reference_update_frequency)
    if args.gamma0 is not None:
        config = replace(config, gamma0=args.gamma0)
    if args.gamma_exponent is not None:
        config = replace(config, gamma_exponent=args.gamma_exponent)
    if args.disable_gradient_normalization:
        config = replace(config, normalize_gradient_update=False)
    if args.gradient_smoothing_window is not None:
        config = replace(config, gradient_smoothing_window=args.gradient_smoothing_window)
    if args.policy_noise_scale is not None:
        config = replace(config, policy_noise_scale=args.policy_noise_scale)
    if args.policy_temperature is not None:
        config = replace(config, policy_temperature=args.policy_temperature)
    if args.policy_normalizer is not None:
        config = replace(config, policy_normalizer=args.policy_normalizer)
    if args.fixed_asset_count is not None:
        config = replace(config, fixed_asset_count=args.fixed_asset_count)
    if args.bootstrap_asset_count is not None:
        config = replace(config, bootstrap_asset_count=args.bootstrap_asset_count)
    if args.dirichlet_execution_mode is not None:
        config = replace(config, dirichlet_execution_mode=args.dirichlet_execution_mode)
    runner = ExperimentRunner(config)
    artifacts = runner.run(methods=args.methods, seeds=args.seeds, dry_run_days=args.dry_run_days)
    print(f"result_dir: {artifacts.result_dir}")
    print(f"trace: {artifacts.trace_path}")
    print(f"summary_by_run: {artifacts.summary_path}")
    print(f"summary_by_method: {artifacts.aggregate_path}")
    print(f"stock_info_source_status: {artifacts.source_status_path}")
    for plot_path in artifacts.plot_paths:
        print(f"plot: {plot_path}")

if __name__ == "__main__":
    main()
