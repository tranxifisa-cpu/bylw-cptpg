from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvp_cpt_pg.config import ExperimentConfig
from mvp_cpt_pg.synthetic_env import (
    SyntheticCPTPGTracker,
    SyntheticMarket,
    SyntheticMarketConfig,
    save_synthetic_method_comparison_plots,
    save_synthetic_per_method_plots,
    save_synthetic_reference_comparison_plot,
    save_synthetic_seed_aggregate_plots,
    save_synthetic_plots,
)
from mvp_cpt_pg.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic non-stationary CPT-PG tracking diagnostics.")
    parser.add_argument("--policy-normalizer", choices=["softmax", "sparsemax", "dirichlet"], required=True)
    parser.add_argument(
        "--estimation-mode",
        choices=["rolling_window", "on_policy_episode"],
        default="rolling_window",
        help="rolling_window replays past h days; on_policy_episode runs the current policy for h days before each update",
    )
    parser.add_argument(
        "--method",
        choices=["dynamic_cpt_pg", "symmetric_cpt_pg", "static_cpt_pg", "expected_return_pg", "exponential_utility_pg"],
        default="dynamic_cpt_pg",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["dynamic_cpt_pg", "symmetric_cpt_pg", "static_cpt_pg", "expected_return_pg", "exponential_utility_pg"],
        default=None,
        help="Run multiple synthetic methods in one output directory",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--start-date", default="20230101")
    parser.add_argument("--end-date", default="20260531")
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--prewarm-steps", type=int, default=30)
    parser.add_argument("--risky-assets", type=int, default=2)
    parser.add_argument("--evaluation-horizon", type=int, default=15)
    parser.add_argument("--cpt-samples", type=int, default=256)
    parser.add_argument("--gradient-samples", type=int, default=32)
    parser.add_argument(
        "--shared-cpt-gradient-samples",
        action="store_true",
        help="Use the same trajectories for CPT quantiles/objective and score-function gradient estimation",
    )
    parser.add_argument("--gamma0", type=float, default=1.0)
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
    parser.add_argument("--policy-noise-scale", type=float, default=0.1)
    parser.add_argument("--policy-temperature", type=float, default=2.0)
    parser.add_argument(
        "--dirichlet-execution-mode",
        choices=["sample", "mean"],
        default="mean",
        help="Dirichlet mode for actual episode execution; gradient/objective estimation always samples",
    )
    parser.add_argument("--eta-gain", type=float, default=0.40)
    parser.add_argument("--eta-loss", type=float, default=0.10)
    parser.add_argument(
        "--reference-update-frequency",
        choices=["daily", "episode"],
        default="daily",
        help="daily updates the dynamic reference after each trading day; episode updates it once after each h-day episode",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--return-bound", type=float, default=0.08)
    parser.add_argument(
        "--factor-strength",
        type=float,
        default=2.5,
        help="Multiplier on semi-synthetic factor return components",
    )
    parser.add_argument("--static-reference", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/synthetic"))
    parser.add_argument("--compare-dynamic-run", type=Path, default=None)
    parser.add_argument("--compare-static-run", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare_dynamic_run is not None or args.compare_static_run is not None:
        if args.compare_dynamic_run is None or args.compare_static_run is None:
            raise RuntimeError("Both --compare-dynamic-run and --compare-static-run are required for reference comparison")
        dynamic_trace = _load_seed_traces(args.compare_dynamic_run)
        static_trace = _load_seed_traces(args.compare_static_run)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_dir(args.output_root / f"comparison_{stamp}_{args.policy_normalizer}")
        path = save_synthetic_reference_comparison_plot(
            dynamic_trace,
            static_trace,
            output_dir / "dynamic_vs_static_seed_quantiles.png",
            policy_label=args.policy_normalizer,
        )
        print(f"comparison_plot: {path}")
        return

    experiment_config = ExperimentConfig(
        evaluation_horizon=args.evaluation_horizon,
        cpt_sample_base=args.cpt_samples,
        gradient_sample_base=args.gradient_samples,
        shared_cpt_gradient_samples=args.shared_cpt_gradient_samples,
        fixed_sample_counts=True,
        gamma0=args.gamma0,
        gamma_exponent=0.0,
        normalize_gradient_update=not args.disable_gradient_normalization,
        gradient_smoothing_window=args.gradient_smoothing_window,
        policy_normalizer=args.policy_normalizer,
        policy_noise_scale=args.policy_noise_scale,
        policy_temperature=args.policy_temperature,
        dirichlet_execution_mode=args.dirichlet_execution_mode,
        eta_gain=args.eta_gain,
        eta_loss=args.eta_loss,
        reference_update_frequency=args.reference_update_frequency,
        trade_cost_bps=args.transaction_cost_bps,
    )
    if args.prewarm_steps < args.evaluation_horizon + 1:
        raise RuntimeError("prewarm_steps must be greater than evaluation_horizon so the first update has a full history window")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    methods = args.methods if args.methods is not None else [args.method]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_label = "multi_method" if len(methods) > 1 else methods[0]
    output_root = ensure_dir(args.output_root / f"run_{stamp}_semi_hs300_{args.policy_normalizer}_{method_label}")
    summary_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    written_outputs: list[Path] = []
    market_by_seed: dict[int, tuple[SyntheticMarketConfig, SyntheticMarket]] = {}

    for method in methods:
        for seed in seeds:
            if seed not in market_by_seed:
                seed_market_config = SyntheticMarketConfig(
                    steps=args.steps,
                    prewarm_steps=args.prewarm_steps,
                    risky_assets=args.risky_assets,
                    seed=seed,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    index_code=args.index_code,
                    return_bound=args.return_bound,
                    transaction_cost_bps=args.transaction_cost_bps,
                    factor_strength=args.factor_strength,
                )
                market_by_seed[seed] = (seed_market_config, SyntheticMarket.build(seed_market_config))
            seed_market_config, market = market_by_seed[seed]
            tracker = SyntheticCPTPGTracker(
                market=market,
                config=experiment_config,
                policy=args.policy_normalizer,
                seed=seed,
                method=method,
                static_reference=args.static_reference,
                dirichlet_execution_mode=args.dirichlet_execution_mode,
                estimation_mode=args.estimation_mode,
            )
            trace = tracker.run()
            trace_frames.append(trace)

            output_dir = ensure_dir(output_root / method / f"seed{seed}")
            trace_path = output_dir / "synthetic_trace.csv"
            trace.to_csv(trace_path, index=False, encoding="utf-8-sig")
            total_realized_gain_count = int(trace["realized_gain_count"].sum())
            total_realized_loss_count = int(trace["realized_loss_count"].sum())
            total_paper_gain_count = int(trace["paper_gain_count"].sum())
            total_paper_loss_count = int(trace["paper_loss_count"].sum())
            aggregate_pgr_denominator = total_realized_gain_count + total_paper_gain_count
            aggregate_plr_denominator = total_realized_loss_count + total_paper_loss_count
            aggregate_pgr = (
                0.0
                if aggregate_pgr_denominator == 0
                else float(total_realized_gain_count / aggregate_pgr_denominator)
            )
            aggregate_plr = (
                0.0
                if aggregate_plr_denominator == 0
                else float(total_realized_loss_count / aggregate_plr_denominator)
            )

            summary = pd.DataFrame(
                [
                    {
                        "policy": args.policy_normalizer,
                        "method": method,
                        "seed": seed,
                        "static_reference": int(args.static_reference),
                        "market_source": "semi_hs300",
                        "asset_count": len(market.asset_codes) - 1,
                        "steps": args.steps,
                        "final_wealth": float(trace["wealth"].iloc[-1]),
                        "mean_cash_weight": float(trace["cash_weight"].mean()),
                        "mean_turnover": float(trace["turnover"].mean()),
                        "mean_pgr": float(trace["pgr"].mean()),
                        "mean_plr": float(trace["plr"].mean()),
                        "mean_disposition_spread": float(trace["disposition_spread"].mean()),
                        "aggregate_pgr": aggregate_pgr,
                        "aggregate_plr": aggregate_plr,
                        "aggregate_disposition_spread": float(aggregate_pgr - aggregate_plr),
                        "total_realized_gain_count": total_realized_gain_count,
                        "total_realized_loss_count": total_realized_loss_count,
                        "total_paper_gain_count": total_paper_gain_count,
                        "total_paper_loss_count": total_paper_loss_count,
                        "final_reference_point": float(trace["reference_point"].iloc[-1]),
                        "final_normalized_wealth": float(trace["normalized_wealth"].iloc[-1]),
                        "mean_relative_wealth": float(trace["relative_wealth"].mean()),
                        "mean_gradient_norm": float(trace["gradient_norm"].mean()),
                        "mean_dynamic_local_regret_term": float(trace["dynamic_local_regret_term"].mean()),
                        "final_average_dynamic_local_regret": float(trace["average_dynamic_local_regret"].iloc[-1]),
                        "final_dynamic_local_regret": float(trace["dynamic_local_regret"].iloc[-1]),
                        "mean_squared_gradient_norm": float(trace["squared_gradient_norm"].mean()),
                        "final_average_squared_gradient_norm": float(trace["average_squared_gradient_norm"].iloc[-1]),
                        "final_cumulative_squared_gradient_norm": float(trace["cumulative_squared_gradient_norm"].iloc[-1]),
                        "mean_objective_estimate": float(trace["objective_estimate"].mean()),
                        "mean_reference_drift": float(trace["reference_drift"].mean()),
                        "mean_window_approximation_proxy": float(trace["window_approximation_proxy"].mean()),
                    }
                ]
            )
            summary_path = output_dir / "synthetic_summary.csv"
            summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
            summary_frames.append(summary)

            config_snapshot = {
                "market_config": asdict(seed_market_config),
                "experiment_config": {
                    "evaluation_horizon": experiment_config.evaluation_horizon,
                    "cpt_sample_base": experiment_config.cpt_sample_base,
                    "gradient_sample_base": experiment_config.gradient_sample_base,
                    "shared_cpt_gradient_samples": experiment_config.shared_cpt_gradient_samples,
                    "fixed_sample_counts": experiment_config.fixed_sample_counts,
                    "gamma0": experiment_config.gamma0,
                    "gamma_exponent": experiment_config.gamma_exponent,
                    "normalize_gradient_update": experiment_config.normalize_gradient_update,
                    "policy_normalizer": experiment_config.policy_normalizer,
                        "method": method,
                        "estimation_mode": args.estimation_mode,
                        "policy_noise_scale": experiment_config.policy_noise_scale,
                    "policy_temperature": experiment_config.policy_temperature,
                    "dirichlet_execution_mode": experiment_config.dirichlet_execution_mode,
                    "eta_gain": experiment_config.eta_gain,
                    "eta_loss": experiment_config.eta_loss,
                    "reference_update_frequency": experiment_config.reference_update_frequency,
                    "trade_cost_bps": experiment_config.trade_cost_bps,
                },
            }
            config_path = output_dir / "config_snapshot.json"
            config_path.write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

            plot_paths = save_synthetic_plots(trace, output_dir)
            written_outputs.extend([trace_path, summary_path, config_path, *plot_paths])

    merged_summary = pd.concat(summary_frames, ignore_index=True)
    merged_summary_path = output_root / "synthetic_summary_by_seed.csv"
    merged_summary.to_csv(merged_summary_path, index=False, encoding="utf-8-sig")
    numeric_columns = merged_summary.select_dtypes(include="number").columns.drop(["seed"], errors="ignore")
    aggregate_summary = merged_summary.groupby(["policy", "method", "static_reference"], as_index=False)[list(numeric_columns)].agg(
        ["mean", "std", "median"]
    )
    aggregate_summary.columns = ["_".join(str(part) for part in col if part != "") for col in aggregate_summary.columns.to_flat_index()]
    aggregate_summary_path = output_root / "synthetic_summary_aggregate.csv"
    aggregate_summary.to_csv(aggregate_summary_path, index=False, encoding="utf-8-sig")
    aggregate_plot_paths = save_synthetic_seed_aggregate_plots(
        trace_frames,
        output_root,
        label=f"semi_hs300_{args.policy_normalizer}",
    )
    if len(methods) > 1:
        aggregate_plot_paths.extend(save_synthetic_method_comparison_plots(trace_frames, output_root))
        aggregate_plot_paths.extend(save_synthetic_per_method_plots(trace_frames, output_root))
    print(f"output_root: {output_root}")
    print(f"summary_by_seed: {merged_summary_path}")
    print(f"summary_aggregate: {aggregate_summary_path}")
    for path in aggregate_plot_paths:
        print(f"aggregate_plot: {path}")
    print(f"runs: {len(seeds) * len(methods)}")
    for path in written_outputs[:12]:
        print(f"output: {path}")
    if len(written_outputs) > 12:
        print(f"output: ... {len(written_outputs) - 12} more files")


def _load_seed_traces(run_dir: Path) -> pd.DataFrame:
    traces: list[pd.DataFrame] = []
    for path in sorted(run_dir.glob("**/seed*/synthetic_trace.csv")):
        frame = pd.read_csv(path)
        seed_text = path.parent.name.replace("seed", "")
        try:
            seed = int(seed_text)
        except ValueError:
            seed = -1
        frame["seed"] = seed
        traces.append(frame)
    if not traces:
        raise RuntimeError(f"No seed traces found under {run_dir}")
    return pd.concat(traces, ignore_index=True)


if __name__ == "__main__":
    main()
