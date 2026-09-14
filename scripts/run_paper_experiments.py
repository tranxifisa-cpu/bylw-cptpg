"""Prepare cached data, run paper experiments 1-5, and plot saved measurements."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from mvp_cpt_pg.controlled_cpt import expected_cost
from mvp_cpt_pg.paper_market import PaperMarket, PaperPanel, file_hash, prepare_panel
from mvp_cpt_pg.paper_experiments import (EpisodeLaw, METHODS, ESTIMATORS, PaperConfig, evaluate,
    multilevel, plugin, rng_for, run_online, summarize, update_reference)
from mvp_cpt_pg.paper_estimator_diagnostics import run_diagnostics
from mvp_cpt_pg.paper_figures import plot_experiment


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-panel", action="store_true")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/cache/raw/tushare")
    parser.add_argument("--panel", type=Path, default=ROOT / "artifacts/inputs/paper_hs300.csv")
    parser.add_argument("--start-date", default="20230301")
    parser.add_argument("--end-date", default="20260529")
    parser.add_argument("--assets", type=int, default=30)
    parser.add_argument("--pool-seed", type=int, default=2022)
    parser.add_argument("--allow-suspension-carry", action="store_true",
                        help="Keep fixed-pool assets on audited missing-quote days: zero return, stale factors, frozen weight")
    parser.add_argument("--experiment", type=int, choices=range(1, 6))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[29, 147, 3141, 42, 3407, 592, 7, 101, 2024, 2025])
    parser.add_argument("--methods", nargs="+", choices=METHODS + ["plugin", "frozen"], default=METHODS)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=.05)
    parser.add_argument("--a0", type=float, default=1)
    parser.add_argument("--theta-radius", type=float, default=2)
    parser.add_argument("--eta-gain", type=float, default=.2)
    parser.add_argument("--eta-loss", type=float, default=.05)
    parser.add_argument("--cost", type=float, default=.001)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--cap", type=int, default=5)
    parser.add_argument("--base", type=int, default=2)
    parser.add_argument("--layer-exponent", type=float, default=1.5)
    parser.add_argument("--estimator", choices=ESTIMATORS, default="multilevel")
    parser.add_argument("--outer-batch", type=int, default=8)
    parser.add_argument("--trajectory-budget", type=int, default=0,
                        help="Expected training trajectory budget per update; zero keeps legacy M/n")
    parser.add_argument("--estimator-diagnostics", action="store_true",
                        help="Experiment 2: fixed-context, cost-matched comparisons in four regimes")
    parser.add_argument("--diagnostic-days", nargs="+", type=int, default=[0, 150, 290, 350])
    parser.add_argument("--budget-grid", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--compare-estimators", nargs="+", choices=ESTIMATORS, default=ESTIMATORS)
    parser.add_argument("--estimator-comparison", action="store_true",
                        help="Experiment 3: same reference mechanism, estimator variants and frozen control")
    parser.add_argument("--evaluation-n", type=int, default=2048)
    parser.add_argument("--evaluation-m", type=int, default=1024)
    parser.add_argument("--diagnostic-window", type=int, default=5)
    parser.add_argument("--rho", type=float, default=.9)
    parser.add_argument("--factor-strength", type=float, default=2.5)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--n-grid", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--m-grid", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 4, 8, 10])
    parser.add_argument("--reference-batches", type=int, default=8)
    parser.add_argument("--reference-n", type=int, default=16384)
    parser.add_argument("--reference-m", type=int, default=8192)
    parser.add_argument("--eta-grid", nargs="+", type=float, default=[0, .05, .1, .2, .4])
    parser.add_argument("--horizon-grid", nargs="+", type=int, default=[2, 5, 10, 25])
    parser.add_argument("--gamma-grid", nargs="+", type=float, default=[.01, .025, .05, .1])
    parser.add_argument("--selected-config", type=Path,
                        help="Use validation-selected adaptation rates, rejecting overlapping seeds")
    parser.add_argument("--test-start", default="20250102", help="Real replay starts here; earlier data fit the simulator")
    parser.add_argument("--total-return-panel", type=Path,
                        help="Optional audited CSV: trade_date,ts_code,gross_return (close-to-close, dividends included)")
    parser.add_argument("--allow-reference-price-proxy", action="store_true",
                        help="Explicitly allow close/pre_close replay despite corporate actions; NOT total-return backtesting")
    return parser.parse_args()


def config_from_args(args):
    return PaperConfig(horizon=args.horizon, gamma=args.gamma, a0=args.a0,
        theta_radius=args.theta_radius, eta_gain=args.eta_gain, eta_loss=args.eta_loss,
        cost=args.cost, m=args.m, n=args.n, cap=args.cap, base=args.base,
        exponent=args.layer_exponent, evaluation_n=args.evaluation_n,
        evaluation_m=args.evaluation_m, window=args.diagnostic_window, rho=args.rho,
        outer_batch=args.outer_batch, trajectory_budget=args.trajectory_budget)


def save_online(output, episodes, daily):
    output.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(output / "episodes.csv", index=False)
    daily.to_csv(output / "daily.csv", index=False)
    summary = summarize(daily)
    summary.to_csv(output / "summary.csv", index=False)
    numeric = summary.select_dtypes(include="number").columns.drop("seed")
    summary.groupby("method")[numeric].agg(["median", lambda x: x.quantile(.25),
                                          lambda x: x.quantile(.75)]).to_csv(output / "summary_aggregate.csv")
    return summary


def experiment_path(args, config):
    paths = {"gain_first": [1, 1.4, 1.2], "loss_first": [1, .8, 1.2]}
    rows, grid = [], []
    for name, values in paths.items():
        reference = 1.0
        for index, wealth in enumerate(values):
            if index:
                reference = float(update_reference(wealth, reference, config))
            rows.append(dict(path=name, time=index, wealth=wealth, reference=reference))
    for gain, loss in itertools.product(np.linspace(0, .8, 41), repeat=2):
        c = replace(config, eta_gain=float(gain), eta_loss=float(loss))
        terminal = []
        for values in paths.values():
            ref = 1.0
            for value in values[1:]:
                ref = float(update_reference(value, ref, c))
            terminal.append(ref)
        grid.append(dict(eta_gain=gain, eta_loss=loss, reference_gap=terminal[0] - terminal[1]))
    pd.DataFrame(rows).to_csv(args.output / "paths.csv", index=False)
    pd.DataFrame(grid).to_csv(args.output / "path_grid.csv", index=False)


def numerical_reference(law, theta, args, dimension):
    rows = []
    for precision, scale in [("half", 1), ("full", 2)]:
        for batch in range(args.reference_batches):
            c = replace(law.config, evaluation_n=args.reference_n * scale,
                        evaluation_m=args.reference_m * scale)
            j, g = evaluate(replace(law, config=c), theta,
                            rng_for(18001 + dimension, batch, scale), "dynamic")
            rows.append(dict(precision=precision, batch=batch, dimension=dimension, objective=j,
                             **{f"g{i}": value for i, value in enumerate(g)}))
    table = pd.DataFrame(rows)
    table.to_csv(args.output / f"numerical_reference_d{dimension}.csv", index=False)
    full = table[table.precision == "full"][[f"g{i}" for i in range(dimension)]].to_numpy()
    return full.mean(axis=0), full.var(axis=0, ddof=1) / len(full)


def experiment_estimator(args, market, config):
    if args.replications < 2 or args.reference_batches < 2:
        raise ValueError("Variance/reference uncertainty requires at least two repetitions")
    if any(n < config.base or n % config.base or not math.log2(n / config.base).is_integer() for n in args.n_grid):
        raise ValueError("n-grid values must equal base * 2**integer")
    if min(args.m_grid) < 2:
        raise ValueError("M grid must be >= 2")
    bias_rows, variance_rows = [], []
    raw_bias, raw_variance = [], []
    previous = np.r_[1.0, np.zeros(len(market.panel.codes) - 1)]
    for dimension in args.dimensions:
        c = replace(config, dimension=dimension)
        theta = np.full(dimension, .1 / np.sqrt(dimension))
        law = EpisodeLaw(market, 0, 1.0, 1.0, previous, c)
        reference, reference_var = numerical_reference(law, theta, args, dimension)
        if dimension == max(args.dimensions):
            for n in args.n_grid:
                cap = int(math.log2(n / c.base))
                level_law = replace(law, config=replace(c, cap=cap))
                repeats = max(1, round((n + c.m) / expected_cost(cap, c.base, c.exponent)))
                samples = {"plugin": [], "multilevel": []}
                for replication in range(args.replications):
                    g, cost = plugin(law, theta, rng_for(22000 + n, replication, 1), n, c.m)
                    values, costs = multilevel(level_law, theta, rng_for(22000 + n, replication, 2), repeats)
                    for estimator, estimate, calls in [("plugin", g, cost),
                                                        ("multilevel", values.mean(axis=0), costs.sum())]:
                        samples[estimator].append(estimate)
                        raw_bias.append(dict(n=n, cap=cap, estimator=estimator, replication=replication,
                            cost=int(calls), **{f"g{i}": v for i, v in enumerate(estimate)}))
                for estimator, values in samples.items():
                    values = np.asarray(values)
                    bias_rows.append(dict(n=n, estimator=estimator,
                        bias_coordinate=values[:, 0].mean() - reference[0],
                        bias_se=np.sqrt(values[:, 0].var(ddof=1) / len(values) + reference_var[0]),
                        bias_norm=float(np.linalg.norm(values.mean(axis=0) - reference)),
                        reference_variance_trace=float(reference_var.sum())))
        estimates = {m: [] for m in args.m_grid}
        costs_by_m = {m: [] for m in args.m_grid}
        for replication in range(args.replications):
            values, costs = multilevel(law, theta, rng_for(33000 + dimension, replication, 1), max(args.m_grid))
            for m in args.m_grid:
                gradient = values[:m].mean(axis=0)
                estimates[m].append(gradient)
                costs_by_m[m].append(costs[:m].sum())
                raw_variance.append(dict(dimension=dimension, M=m, replication=replication,
                    cost=int(costs[:m].sum()), **{f"g{i}": v for i, v in enumerate(gradient)}))
        for m, values in estimates.items():
            values = np.asarray(values)
            variance_rows.append(dict(dimension=dimension, M=m, cost=np.mean(costs_by_m[m]),
                variance_trace=values.var(axis=0, ddof=1).sum(),
                mse=np.mean(np.sum((values - reference) ** 2, axis=1)),
                reference_variance_trace=float(reference_var.sum())))
        print(f"Finished estimator diagnostics d={dimension}", flush=True)
    for name, rows in [("bias_raw", raw_bias), ("bias_summary", bias_rows),
                        ("variance_raw", raw_variance), ("variance_summary", variance_rows)]:
        pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)


def experiment_ablation(args, market, config):
    if any(args.steps % h for h in args.horizon_grid):
        raise ValueError("Use a common number of days divisible by every horizon-grid value")
    variants = []
    for gain, loss in itertools.product(args.eta_grid, repeat=2):
        variants.append(("eta", replace(config, eta_gain=gain, eta_loss=loss)))
    for horizon in args.horizon_grid:
        # Match expected trajectory-step cost PER MARKET DAY across horizon sweeps.
        variants.append(("horizon", replace(config, horizon=horizon)))
    for gamma in args.gamma_grid:
        variants.append(("gamma", replace(config, gamma=gamma)))
    rows = []
    for index, (sweep, c) in enumerate(variants):
        sub = args.output / f"case_{index:03d}"
        episodes, daily = run_online(market, c, args.seeds, ["dynamic"], args.steps,
                                     estimator=args.estimator)
        summary = save_online(sub, episodes, daily)
        for seed, data in episodes.groupby("seed"):
            met = summary[summary.seed == seed].iloc[0]
            rows.append(dict(case=index, sweep=sweep, seed=seed, eta_gain=c.eta_gain,
                eta_loss=c.eta_loss, horizon=c.horizon, gamma=c.gamma,
                average_dlr=data.average_dynamic_local_regret.iloc[-1],
                mean_q_squared=data.q_squared.mean(), terminal_wealth=met.terminal_wealth,
                max_drawdown=met.max_drawdown, sharpe=met.sharpe,
                training_steps=data.training_steps.sum()))
        pd.DataFrame(rows).to_csv(args.output / "sensitivity.csv", index=False)
    table = pd.DataFrame(rows)
    # Selection never uses cross-target gradient magnitude: it can reward a flat objective.
    candidates = table[(table.sweep == "eta") & (table.eta_gain > table.eta_loss) &
                       (table.eta_loss > 0)].copy()
    candidates["selection_score"] = np.log(candidates.terminal_wealth) - candidates.max_drawdown
    ranking = candidates.groupby(["eta_gain", "eta_loss"], as_index=False).selection_score.median()
    ranking = ranking.sort_values(["selection_score", "eta_gain", "eta_loss"], ascending=[False, True, True])
    if ranking.empty:
        raise ValueError("No positive asymmetric adaptation pair in the validation grid")
    best = ranking.iloc[0]
    ranking.to_csv(args.output / "validation_ranking.csv", index=False)
    selected = dict(eta_gain=float(best.eta_gain), eta_loss=float(best.eta_loss),
                    validation_seeds=args.seeds,
                    validation_last_date=str(market.panel.dates[args.steps - 1]),
                    selection_rule="median(log(terminal_wealth) - maximum_drawdown); positive asymmetric pairs only",
                    config=asdict(config), panel_sha256=file_hash(args.panel),
                    warning="Economic validation choice, not proof of universal optimality; retain the full grid")
    (args.output / "selected_config.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


def main():
    args = arguments()
    if args.prepare_panel:
        metadata = prepare_panel(args.cache, args.panel, args.start_date, args.end_date,
                                 args.assets, args.pool_seed, args.allow_suspension_carry)
        print(f"Prepared {len(metadata['assets'])} stocks: {args.panel}")
        return
    if args.experiment is None or args.output is None:
        raise ValueError("--experiment and --output are required")
    if args.estimator_diagnostics and args.experiment != 2:
        raise ValueError("--estimator-diagnostics requires experiment 2")
    if args.estimator_comparison and args.experiment != 3:
        raise ValueError("--estimator-comparison requires experiment 3")
    if len(set(args.compare_estimators)) != len(args.compare_estimators):
        raise ValueError("Compared estimators must be distinct")
    if args.estimator_comparison and (args.trajectory_budget <= args.n or args.n != args.base * 2**args.cap):
        raise ValueError("Comparison requires trajectory-budget > n and n = base * 2**cap")
    if args.plot_only:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["experiment"] != args.experiment or manifest["status"] != "complete":
            raise ValueError("Wrong experiment or incomplete output")
        plot_experiment(args.output, args.experiment)
        return
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Seeds must be distinct nonnegative integers")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}; use --plot-only or a new directory")
    config = config_from_args(args)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.selected_config:
        selected = json.loads(args.selected_config.read_text(encoding="utf-8"))
        if set(args.seeds).intersection(selected["validation_seeds"]):
            raise ValueError("Validation and test seeds overlap")
        if args.experiment != 1 and file_hash(args.panel) != selected["panel_sha256"]:
            raise ValueError("Validation/test panels differ; do not silently transfer the selection protocol")
        if args.experiment == 5 and args.test_start <= selected["validation_last_date"]:
            raise ValueError("Real test start must follow all validation factor dates")
        config = replace(config, eta_gain=selected["eta_gain"], eta_loss=selected["eta_loss"])
    market, start_day = None, 0
    if args.experiment != 1:
        panel = PaperPanel.load(args.panel, args.start_date, args.end_date)
        if config.horizon > len(panel.dates):
            raise ValueError("Horizon exceeds available factor dates")
        if args.experiment == 5:
            if args.total_return_panel:
                total = pd.read_csv(args.total_return_panel, dtype={"trade_date": str, "ts_code": str})
                gross = total.pivot(index="trade_date", columns="ts_code", values="gross_return")
                values = gross.reindex(index=panel.dates, columns=panel.codes[1:]).to_numpy(float)
                if not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError("Audited total-return panel must cover all dates/assets with positive gross returns")
                panel.returns = np.c_[np.zeros(len(values)), values - 1]
            # Decide from t-2 close, submit before t-1 closing auction, then earn t's relative.
            # Using t-1 close factors to fill at that same close would leak information.
            panel = PaperPanel(panel.dates[1:], panel.codes, panel.features[:-1],
                               panel.returns[1:], panel.corporate_break[1:])
        market = PaperMarket(panel, semi=args.experiment != 5, factor_strength=args.factor_strength)
        if args.experiment == 5:
            start_day = int(np.searchsorted(panel.dates, args.test_start))
            if start_day <= config.horizon or start_day >= len(panel.dates):
                raise ValueError("Need a valid test-start after historical simulator warmup")
            if panel.corporate_break.any() and not args.allow_reference_price_proxy and not args.total_return_panel:
                raise ValueError("Corporate-action reference-price breaks detected. Supply audited total-return data "
                                 "or explicitly use --allow-reference-price-proxy for a price-relative experiment only.")
        if args.experiment in (3, 4, 5) and start_day + args.steps > len(panel.dates):
            raise ValueError(f"Only {len(panel.dates) - start_day} evaluation days available; reduce --steps")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    sources = [Path(__file__), ROOT / "mvp_cpt_pg/paper_market.py", ROOT / "mvp_cpt_pg/paper_experiments.py",
               ROOT / "mvp_cpt_pg/paper_figures.py", ROOT / "mvp_cpt_pg/controlled_cpt.py",
               ROOT / "mvp_cpt_pg/synthetic_env.py", ROOT / "mvp_cpt_pg/paper_estimator_diagnostics.py"]
    manifest = dict(experiment=args.experiment, status="running", config=asdict(config),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        started_utc=datetime.now(timezone.utc).isoformat(),
        panel_sha256=file_hash(args.panel) if market else None,
        cost_model="paper: X_next=X*(1+w.R-c*L1(w-w_previous_target)); cash included",
        execution="mean; update after each episode; training trajectories sample",
        diagnostic="independent finite-sample plug-in Q, not exact population residual or certified set distance",
        simulator="prescribed exogenous factor path + conditional synthetic returns" if args.experiment != 5 else
                  "past-only empirical block simulator; execution return basis is declared separately",
        real_timing="t-2 close information; preannounced t-1 closing-auction target; t close-to-close relative",
        return_basis=("conditional synthetic returns" if args.experiment != 5 else
                      "externally supplied audited total returns" if args.total_return_panel else "quote reference-price relatives"),
        total_return_sha256=file_hash(args.total_return_panel) if args.total_return_panel else None,
        estimator_notes="finite capped multilevel has residual bias; QQ is standardized finite-cap data; no minimax lower bound inferred")
    if args.estimator_diagnostics or args.estimator_comparison:
        manifest["comparison_estimators"] = args.compare_estimators
        manifest["budget_basis"] = "Full trajectories; repeats=floor(budget/expected_cost); actual costs saved"
        manifest["finite_target"] = "n=base*2**cap; all compared estimators have the same finite-level expectation"
    if args.estimator_comparison:
        manifest["common_probe"] = "Same day, initial wealth=reference=1, all cash, same reference rates; saved pre-update theta"
        manifest["common_gradient_cross"] = "g1 dot g2 estimates squared mean finite-sample gradient, not exact population residual; negative samples retained"
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        if args.experiment == 1:
            experiment_path(args, config)
        elif args.experiment == 2:
            if args.estimator_diagnostics:
                run_diagnostics(args, market, config)
            else:
                experiment_estimator(args, market, config)
        elif args.experiment == 4:
            experiment_ablation(args, market, config)
        elif args.estimator_comparison:
            all_episodes, all_daily = [], []
            for variant in [*args.compare_estimators, "frozen"]:
                episodes, daily = run_online(market, config, args.seeds,
                    ["frozen" if variant == "frozen" else "dynamic"], args.steps, start_day,
                    "multilevel" if variant == "frozen" else variant, common_probe=True)
                episodes["method"], daily["method"] = variant, variant
                all_episodes.append(episodes)
                all_daily.append(daily)
                save_online(args.output, pd.concat(all_episodes), pd.concat(all_daily))
        else:
            episodes, daily = run_online(market, config, args.seeds, args.methods,
                args.steps, start_day, args.estimator)
            save_online(args.output, episodes, daily)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_experiment(args.output, args.experiment)
    print(f"Measurements and PDF/SVG/PNG figures: {args.output}")


if __name__ == "__main__":
    main()
