"""Run the paper's external-validity experiments 5--7.

Examples are kept in experiments/external_validity/README.md.  Each
experiment writes a self-contained manifest and refuses to overwrite output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from mvp_cpt_pg.external_validity import (build_e6_opportunities, fit_agent_types,
    fit_e6_models, plot_e6, plot_e7, replay_e5_recommendations, run_e5,
    simulate_agents)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=int, choices=(5, 6, 7), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=ROOT / "artifacts/cache/raw/tushare")
    p.add_argument("--behavior-root", type=Path, default=ROOT / "behavior")
    p.add_argument("--panel", type=Path, default=ROOT / "artifacts/inputs/paper_hs300_full.csv")
    p.add_argument("--e5-run", type=Path,
                   help="E7 input: completed E5 run directory containing manifest/daily/episodes")
    p.add_argument("--start-date", default="20230103")
    p.add_argument("--test-start", default="20250102")
    p.add_argument("--end-date", default="20260529")
    p.add_argument("--assets", type=int, default=300)
    p.add_argument("--pool-seed", type=int, default=2022)
    p.add_argument("--seeds", nargs="+", type=int, default=[29, 147, 3141, 42, 3407, 592, 7, 101, 2024, 2025])
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--gamma", type=float,
                   help="E5 policy step; defaults to the formal value 0.08")
    p.add_argument("--trajectory-budget", type=int,
                   help="E5 expected training budget; defaults to the formal value 512")
    p.add_argument("--evaluation-n", type=int, default=512)
    p.add_argument("--evaluation-m", type=int, default=256)
    p.add_argument("--eta-gain", type=float,
                   help="Reference gain speed; defaults to 0.20 for E5 and 0.40 for E6")
    p.add_argument("--eta-loss", type=float,
                   help="Reference loss speed; defaults to 0.05 for E5 and 0.10 for E6")
    p.add_argument("--cost", type=float, default=.001)
    p.add_argument("--trade-fraction", type=float, default=.2)
    p.add_argument("--policy-sharpness", type=float,
                   help="E5 inverse temperature beta; defaults to the formal value 6")
    p.add_argument("--dirichlet-scale", type=float,
                   help="E5 concentration c0 in alpha_i=c0*exp(beta*z_i); defaults to 10")
    p.add_argument("--allow-suspension-carry", action="store_true", default=True)
    p.add_argument("--max-opportunities", type=int, default=0,
                   help="E6 pilot cap; zero reconstructs all eligible opportunities")
    p.add_argument("--e6-opportunities-input", type=Path,
                   help="E6 input: reuse an existing opportunities.parquet instead of rebuilding events")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--agents-per-type", type=int, default=16)
    p.add_argument("--e6-shrinkage", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--e6-opportunities", type=Path,
                   help="E7 input: existing E6 opportunities; omit to rebuild from behavior into E7 output")
    p.add_argument("--e6-model", type=Path,
                   help="E7 input: E6 model directory containing metrics.csv")
    return p.parse_args()


def main():
    args = parser()
    if args.experiment == 5:
        gamma = .08 if args.gamma is None else args.gamma
        trajectory_budget = 512 if args.trajectory_budget is None else args.trajectory_budget
        eta_gain = .2 if args.eta_gain is None else args.eta_gain
        eta_loss = .05 if args.eta_loss is None else args.eta_loss
        policy_sharpness = 6.0 if args.policy_sharpness is None else args.policy_sharpness
        dirichlet_scale = 10.0 if args.dirichlet_scale is None else args.dirichlet_scale
        summary = run_e5(cache_root=args.cache, panel_path=args.panel, output=args.output,
                         start_date=args.start_date, test_start=args.test_start, end_date=args.end_date,
                         assets=args.assets, pool_seed=args.pool_seed, seeds=args.seeds,
                         horizon=args.horizon, gamma=gamma, trajectory_budget=trajectory_budget,
                         evaluation_n=args.evaluation_n, evaluation_m=args.evaluation_m,
                         eta_gain=eta_gain, eta_loss=eta_loss, cost=args.cost,
                         trade_fraction=args.trade_fraction,
                         policy_sharpness=policy_sharpness,
                         dirichlet_scale=dirichlet_scale,
                         allow_suspension_carry=args.allow_suspension_carry)
        print(summary.to_string(index=False))
        return
    if args.experiment == 6:
        eta_gain = .4 if args.eta_gain is None else args.eta_gain
        eta_loss = .1 if args.eta_loss is None else args.eta_loss
        args.output.mkdir(parents=True, exist_ok=True)
        opportunities_path = (args.e6_opportunities_input if args.e6_opportunities_input is not None
                              else args.output / "opportunities.parquet")
        if args.e6_opportunities_input is not None and not opportunities_path.exists():
            raise FileNotFoundError(opportunities_path)
        if opportunities_path.exists():
            opportunities = pd.read_parquet(opportunities_path)
            metadata_path = opportunities_path.with_suffix(".json")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                old_gain = float(metadata.get("eta_gain", eta_gain))
                old_loss = float(metadata.get("eta_loss", eta_loss))
                if abs(old_gain - eta_gain) > 1e-12 or abs(old_loss - eta_loss) > 1e-12:
                    raise ValueError(
                        "Existing E6 opportunities use different eta values; use a new --output "
                        "directory before changing --eta-gain/--eta-loss"
                    )
        else:
            opportunities = build_e6_opportunities(args.behavior_root, opportunities_path,
                                                   max_opportunities=args.max_opportunities,
                                                   eta_gain=eta_gain, eta_loss=eta_loss)
        metrics, predictions = fit_e6_models(opportunities, args.output / "model", args.bootstrap, args.seed,
                                             eta_gain=eta_gain, eta_loss=eta_loss,
                                             opportunities_source=opportunities_path)
        plot_e6(metrics, predictions, args.output / "model" / "figures")
        print(metrics[metrics.split == "test"].to_string(index=False))
        return
    if args.e5_run is None:
        raise ValueError("Experiment 7 requires --e5-run pointing to a completed experiment 5 directory")
    if not args.e5_run.exists():
        raise FileNotFoundError(args.e5_run)
    if args.e6_model is None:
        raise ValueError("Experiment 7 requires --e6-model")
    if args.agents_per_type < 1:
        raise ValueError("--agents-per-type must be positive")
    if args.e6_shrinkage <= 0:
        raise ValueError("--e6-shrinkage must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    e6_manifest = json.loads((args.e6_model / "manifest.json").read_text(encoding="utf-8"))
    e6_opportunities = args.e6_opportunities
    if e6_opportunities is None:
        e6_opportunities = args.output / "e6_opportunities.parquet"
        reconstructed = build_e6_opportunities(
            args.behavior_root, e6_opportunities, eta_gain=float(e6_manifest["eta_gain"]),
            eta_loss=float(e6_manifest["eta_loss"]))
        if len(reconstructed) != int(e6_manifest["rows"]):
            raise ValueError("Rebuilt E6 opportunities do not match the completed E6 model")
    elif not e6_opportunities.exists():
        raise FileNotFoundError(e6_opportunities)
    else:
        source_metadata = e6_opportunities.with_suffix(".json")
        if not source_metadata.exists():
            raise ValueError("An explicit E6 opportunities input needs its matching .json metadata")
        source_config = json.loads(source_metadata.read_text(encoding="utf-8"))
        if (float(source_config["eta_gain"]) != float(e6_manifest["eta_gain"]) or
                float(source_config["eta_loss"]) != float(e6_manifest["eta_loss"])):
            raise ValueError("E6 opportunities and model use different reference speeds")
        if len(pd.read_parquet(e6_opportunities, columns=["y"])) != int(e6_manifest["rows"]):
            raise ValueError("E6 opportunities row count does not match the completed E6 model")
    recommendations, panel, config, e5_metadata = replay_e5_recommendations(
        args.e5_run, args.output / "e5_recommendations.parquet")
    assignments, profiles = fit_agent_types(args.behavior_root, args.output, args.clusters, args.seed,
                                            e6_opportunities=e6_opportunities,
                                            e6_model=args.e6_model,
                                            shrinkage=args.e6_shrinkage)
    summary, traces, overall = simulate_agents(
        recommendations, assignments, profiles, panel, config, args.output,
        investor_eta_gain=float(e6_manifest["eta_gain"]),
        investor_eta_loss=float(e6_manifest["eta_loss"]), seed=args.seed,
        agents_per_type=args.agents_per_type, source_metadata=e5_metadata)
    plot_e7(summary, pd.read_csv(args.output / "agent_type_profiles_simulated.csv"),
            args.output / "figures", overall=overall)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
