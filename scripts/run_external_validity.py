"""Run the paper's external-validity experiments 5--7.

Examples are kept in experiments/external_validity/README.md.  Each
experiment writes a self-contained manifest and refuses to overwrite output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from mvp_cpt_pg.external_validity import (build_e6_opportunities, fit_agent_types,
    fit_e6_models, plot_e6, plot_e7, run_e5, simulate_agents)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=int, choices=(5, 6, 7), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=ROOT / "artifacts/cache/raw/tushare")
    p.add_argument("--behavior-root", type=Path, default=ROOT / "behavior")
    p.add_argument("--panel", type=Path, default=ROOT / "artifacts/inputs/e5_hs300_full.csv")
    p.add_argument("--e5-daily", type=Path)
    p.add_argument("--start-date", default="20230103")
    p.add_argument("--test-start", default="20250102")
    p.add_argument("--end-date", default="20260529")
    p.add_argument("--assets", type=int, default=300)
    p.add_argument("--pool-seed", type=int, default=2022)
    p.add_argument("--seeds", nargs="+", type=int, default=[29, 147, 3141, 42, 3407, 592, 7, 101, 2024, 2025])
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--gamma", type=float, default=.05)
    p.add_argument("--trajectory-budget", type=int, default=256)
    p.add_argument("--evaluation-n", type=int, default=512)
    p.add_argument("--evaluation-m", type=int, default=256)
    p.add_argument("--eta-gain", type=float, default=.2)
    p.add_argument("--eta-loss", type=float, default=.05)
    p.add_argument("--cost", type=float, default=.001)
    p.add_argument("--trade-fraction", type=float, default=.2)
    p.add_argument("--allow-suspension-carry", action="store_true", default=True)
    p.add_argument("--max-opportunities", type=int, default=0,
                   help="E6 pilot cap; zero reconstructs all eligible opportunities")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main():
    args = parser()
    if args.experiment == 5:
        summary = run_e5(cache_root=args.cache, panel_path=args.panel, output=args.output,
                         start_date=args.start_date, test_start=args.test_start, end_date=args.end_date,
                         assets=args.assets, pool_seed=args.pool_seed, seeds=args.seeds,
                         horizon=args.horizon, gamma=args.gamma, trajectory_budget=args.trajectory_budget,
                         evaluation_n=args.evaluation_n, evaluation_m=args.evaluation_m,
                         eta_gain=args.eta_gain, eta_loss=args.eta_loss, cost=args.cost,
                         trade_fraction=args.trade_fraction, allow_suspension_carry=args.allow_suspension_carry)
        print(summary.to_string(index=False))
        return
    if args.experiment == 6:
        args.output.mkdir(parents=True, exist_ok=True)
        opportunities_path = args.output / "opportunities.parquet"
        if opportunities_path.exists():
            opportunities = pd.read_parquet(opportunities_path)
        else:
            opportunities = build_e6_opportunities(args.behavior_root, opportunities_path,
                                                   max_opportunities=args.max_opportunities)
        metrics, predictions = fit_e6_models(opportunities, args.output / "model", args.bootstrap, args.seed)
        plot_e6(metrics, predictions, args.output / "model" / "figures")
        print(metrics[metrics.split == "test"].to_string(index=False))
        return
    if args.e5_daily is None:
        raise ValueError("Experiment 7 requires --e5-daily pointing to experiment 5/daily.csv")
    if not args.e5_daily.exists():
        raise FileNotFoundError(args.e5_daily)
    args.output.mkdir(parents=True, exist_ok=True)
    assignments, profiles = fit_agent_types(args.behavior_root, args.output, args.clusters, args.seed)
    summary, traces = simulate_agents(pd.read_csv(args.e5_daily), assignments, profiles, args.output, args.seed)
    plot_e7(summary, pd.read_csv(args.output / "agent_type_profiles_simulated.csv"), args.output / "figures")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
