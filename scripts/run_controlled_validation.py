"""Run claim-linked controlled experiments without changing market backtests."""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from mvp_cpt_pg.controlled_cpt import (
    BetaMarket, SmoothCPT, capped_estimates, expected_cost, paired_quantile_weight,
    plugin_estimate, projected_update, residual, residual_metrics, stationary_points,
)


def rng_for(seed, *parts):
    digest = hashlib.sha256(json.dumps([seed, *parts]).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def error_summary(values, truth):
    values = np.asarray(values)
    loss = (values - truth) ** 2
    return dict(mean=float(values.mean()), bias=float(values.mean() - truth),
                mean_se=float(values.std(ddof=1) / np.sqrt(len(values))),
                variance=float(values.var(ddof=1)), mse=float(loss.mean()),
                mse_se=float(loss.std(ddof=1) / np.sqrt(len(values))))


def estimator_experiments(args, output):
    model = BetaMarket(probability=1.0, return_size=1 / 3)
    reference_file = ROOT / "output_paper/DRCPT_PG_Third_Optimized/results/full_results.json"
    published = json.loads(reference_file.read_text(encoding="utf-8"))
    truth = published["exact_gradient_quadrature"]
    expectation = truth + published["analytic_bias_coefficient"] / 128
    rng = rng_for(args.seed, "independent_n128")
    outer, score = model.draw(rng, 0, args.bias_repetitions)
    inner, _ = model.draw(rng, 0, (args.bias_repetitions, 128))
    fine = paired_quantile_weight(inner, outer, model.cpt) * score
    half = (paired_quantile_weight(inner[:, :64], outer, model.cpt)
            + paired_quantile_weight(inner[:, 64:], outer, model.cpt)) * score / 2
    write_rows(output / "bias_recheck.csv", [dict(repetition=i, n=128, score=float(score[i]),
               T_fine=float(fine[i]), Delta=float(fine[i] - half[i])) for i in range(len(fine))])
    recheck = error_summary(fine, truth)
    recheck.update(n=128, repetitions=len(fine), analytic_bias=expectation - truth,
                   expected_T=expectation,
                   discrepancy_in_reported_se=(fine.mean() - expectation) / recheck["mean_se"],
                   top_one_percent_absolute_contribution=float(
                       np.sum(fine[np.argsort(np.abs(fine))[-max(1, len(fine) // 100):]]) / len(fine)))
    write_rows(output / "bias_recheck_summary.csv", [recheck])
    print("bias_recheck", json.dumps(recheck), flush=True)

    models = {"pure_gain": model, "mixed_gain_loss": BetaMarket(probability=0.6)}
    rows, summaries, tuning = [], [], []
    for model_name, market in models.items():
        truth = market.oracle(0, 512)[1]
        oracle_difference = abs(truth - market.oracle(0, 1024)[1])
        for budget in args.budgets:
            candidates = []
            for share in (0.25, 0.5, 0.75):
                n = max(1, min(budget - 1, round(budget * share)))
                m = budget - n
                rng = rng_for(args.seed, "allocation_only", model_name, budget, share)
                estimates = [plugin_estimate(market, 0, rng, n, m)
                             for _ in range(args.allocation_repetitions)]
                score = float(np.mean((np.asarray(estimates) - truth) ** 2))
                candidates.append((score, n, m))
                tuning.append(dict(model=model_name, budget=budget, n=n, m=m,
                                   repetitions=len(estimates), validation_mse=score))
            _, n, m = min(candidates)
            for method in ("plugin", "capped_multilevel"):
                rng = rng_for(args.seed, "evaluation_only", model_name, budget, method)
                start = time.perf_counter()
                repeats = max(1, round(budget / expected_cost(args.cap)))
                if method == "plugin":
                    estimates = np.array([plugin_estimate(market, 0, rng, n, m)
                                          for _ in range(args.repetitions)])
                    costs = np.full(args.repetitions, budget)
                    planned_cost = float(budget)
                else:
                    values, actual = capped_estimates(market, 0, rng,
                                                     repeats * args.repetitions, cap=args.cap)
                    estimates = values.reshape(args.repetitions, repeats).mean(axis=1)
                    costs = actual.reshape(args.repetitions, repeats).sum(axis=1)
                    planned_cost = repeats * expected_cost(args.cap)
                elapsed = time.perf_counter() - start
                for i, (value, cost) in enumerate(zip(estimates, costs)):
                    rows.append(dict(model=model_name, nominal_budget=budget, method=method,
                                     repetition=i, estimate=float(value), raw_trajectories=int(cost)))
                summary = dict(model=model_name, nominal_budget=budget, method=method,
                               repetitions=args.repetitions, true_gradient=truth,
                               oracle_order_difference=oracle_difference, theta=0.0,
                               inner_n=n if method == "plugin" else 0,
                               outer_m=m if method == "plugin" else 0,
                               multilevel_M=repeats if method != "plugin" else 0,
                               cap=args.cap if method != "plugin" else -1,
                               expected_raw_cost=planned_cost, average_raw_cost=float(costs.mean()),
                               elapsed_seconds=elapsed, **error_summary(estimates, truth))
                summaries.append(summary)
                print("estimator", model_name, budget, method, "MSE", summary["mse"], flush=True)
    write_rows(output / "estimator_replications.csv", rows)
    write_rows(output / "estimator_summary.csv", summaries)
    write_rows(output / "allocation_selection.csv", tuning)


def path_experiments(output):
    paths = {"up_then_down": [1.0, 1.4, 1.2], "down_then_up": [1.0, 0.8, 1.2]}
    modes = {"asymmetric": (0.4, 0.1), "symmetric": (0.25, 0.25), "static": (0.0, 0.0)}
    rows = []
    for mode, (gain, loss) in modes.items():
        for name, history in paths.items():
            reference = 1.0
            sequence = history + [1.2] * 30
            for day, wealth in enumerate(sequence):
                if day:
                    eta = gain if wealth >= reference else loss
                    reference += eta * (wealth - reference)
                market = BetaMarket(wealth=wealth, reference=reference, eta_gain=gain, eta_loss=loss)
                objective, gradient = market.oracle(0, 256)
                rows.append(dict(mode=mode, path=name, day=day, wealth=wealth, reference=reference,
                                 prospective_cpt=objective, prospective_gradient=gradient,
                                 next_mean_weight=float(1 / (1 + np.exp(-projected_update(0, gradient, 0.05))))))
    write_rows(output / "reference_paths.csv", rows)


def market_probability(scenario, k):
    if scenario == "fixed":
        return 0.7
    if scenario == "fixed_loss":
        return 0.3
    if scenario == "slow":
        return float(0.5 + 0.15 * np.sin(0.2 * np.sqrt(k)))
    if scenario == "switching":
        return 0.3 if ((k - 1) // 100) % 2 else 0.7
    raise ValueError(scenario)


def tracking_experiments(args, output):
    base = BetaMarket()
    cpt = base.cpt
    b_value = float(cpt.loss_aversion * cpt.magnitude((1 - base.eta_loss) * base.return_size))
    c1 = float(cpt.weight_prime(1))
    eps = cpt.epsilon
    c2 = 6 * (1 - 2 * eps)**2 * (1 - eps) / ((1 - eps)**3 - eps**3)
    # Beta score has E[G^2]=1, and E[|G^2 + Hessian(log pi)|] <= 2.
    smoothness_bound = 2 * b_value * (c2 + 2 * c1)
    a0 = 1.0
    gamma = a0 / (2 * smoothness_bound)
    metadata = dict(domain=[-2, 2], a0=a0, gamma=gamma, smoothness_bound=smoothness_bound,
                    gamma_L=gamma * smoothness_bound, window=args.window, rho=args.rho,
                    h=1, fixed_context=True, evaluation_type="known_model_numerical_quadrature",
                    stationarity_set="grid/bracket approximation, not an error-bound certificate")
    (output / "tracking_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    theta_grid = np.linspace(-2, 2, 81)
    grid_rows, traces, summaries = [], [], []
    numerical_sets = {}
    all_probabilities = sorted({market_probability(sc, k) for sc in args.scenarios
                                for k in range(1, args.episodes + 1)})
    for p in all_probabilities:
        market = replace(base, probability=p)
        numerical_sets[p] = stationary_points(market, order=256)
    for p in [0.3, 0.5, 0.6, 0.7]:
        market = replace(base, probability=p)
        js, gs = market.oracle(theta_grid, 256)
        for theta, objective, gradient in zip(theta_grid, js, gs):
            grid_rows.append(dict(probability=p, theta=theta, objective=objective, gradient=gradient))
    write_rows(output / "target_grid.csv", grid_rows)

    for scenario in args.scenarios:
        methods = ["capped_multilevel", "plugin", "oracle", "frozen"]
        if args.increasing_precision and scenario != "switching":
            methods.append("multilevel_increasing")
        for method in methods:
            seeds = args.seeds if method not in ("oracle", "frozen") else args.seeds[:1]
            for seed in seeds:
                rng = rng_for(args.seed, "tracking", scenario, method, seed)
                theta = 0.0
                records = []
                previous_target = None
                accumulated_grid_variation = 0.0
                for k in range(1, args.episodes + 1):
                    p = market_probability(scenario, k)
                    market = replace(base, probability=p)
                    objective, gradient = market.oracle(theta, 256)
                    check_j, check_g = market.oracle(theta, 128)
                    cap = args.cap
                    repeat_count = args.tracking_repeats
                    if method == "multilevel_increasing":
                        repeat_count = math.ceil(8 * math.sqrt(k))
                        cap = 3 + math.ceil(math.log2(k + 1) / 2)
                    budget = max(2, round(repeat_count * expected_cost(cap)))
                    if method.startswith("capped") or method == "multilevel_increasing":
                        estimates, costs = capped_estimates(market, theta, rng, repeat_count, cap=cap)
                        estimated = float(estimates.mean())
                        cost = int(costs.sum())
                    elif method == "plugin":
                        estimated = plugin_estimate(market, theta, rng, budget // 2, budget - budget // 2)
                        cost = budget
                    else:
                        estimated = gradient if method == "oracle" else 0.0
                        cost = 0
                    next_theta = theta if method == "frozen" else projected_update(theta, estimated, gamma, a0)
                    q = residual(theta, gradient, gamma, a0)
                    targets = market.oracle(theta_grid, 128)[0]
                    if previous_target is not None:
                        accumulated_grid_variation += float(np.max(np.abs(targets - previous_target)))
                    previous_target = targets
                    records.append(dict(scenario=scenario, method=method, seed=seed, k=k,
                                        probability=p, theta=theta, next_theta=next_theta,
                                        objective=objective, true_gradient=gradient, estimated_gradient=estimated,
                                        Q=q, squared_gradient_error=(estimated - gradient)**2 if method != "frozen" else "",
                                        stationary_distance=float(np.min(np.abs(numerical_sets[p] - theta))),
                                        stationary_points=json.dumps(numerical_sets[p].tolist()),
                                        raw_trajectories=cost, M=repeat_count if "multilevel" in method else 0,
                                        cap=cap if "multilevel" in method else -1,
                                        oracle_order_difference=max(abs(check_j - objective), abs(check_g - gradient)),
                                        grid_value_variation=accumulated_grid_variation))
                    theta = next_theta
                smooth, dlr, avg_dlr, avg_q = residual_metrics([r["Q"] for r in records], args.window, args.rho)
                for index, record in enumerate(records):
                    record.update(smoothed_Q=float(smooth[index]), dynamic_local_regret=float(dlr[index]),
                                  average_dynamic_local_regret=float(avg_dlr[index]), average_squared_Q=float(avg_q[index]))
                traces.extend(records)
                summaries.append(dict(scenario=scenario, method=method, seed=seed, episodes=args.episodes,
                                      final_theta=theta, average_squared_Q=float(avg_q[-1]),
                                      dynamic_local_regret=float(dlr[-1]), average_dynamic_local_regret=float(avg_dlr[-1]),
                                      mean_stationary_distance=float(np.mean([r["stationary_distance"] for r in records])),
                                      raw_trajectories=sum(r["raw_trajectories"] for r in records),
                                      maximum_oracle_order_difference=max(r["oracle_order_difference"] for r in records)))
                print("tracking", scenario, method, seed, "theta", round(theta, 6),
                      "mean Q2", round(float(avg_q[-1]), 10), flush=True)
    write_rows(output / "tracking.csv", traces)
    write_rows(output / "tracking_summary.csv", summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suite", choices=["all", "estimator", "tracking", "path"], default="all")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--allocation-repetitions", type=int, default=200)
    parser.add_argument("--bias-repetitions", type=int, default=1600)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--cap", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--tracking-repeats", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[29, 147, 592])
    parser.add_argument("--scenarios", nargs="+", choices=["fixed", "fixed_loss", "slow", "switching"],
                        default=["fixed", "slow", "switching"])
    parser.add_argument("--increasing-precision", action="store_true")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--rho", type=float, default=0.9)
    args = parser.parse_args()
    if min(args.repetitions, args.bias_repetitions, args.allocation_repetitions) < 2:
        parser.error("Statistical summaries require at least two replications")
    if min(args.budgets) < 4 or args.episodes < 1 or args.tracking_repeats < 1:
        parser.error("Invalid experiment size")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Duplicate learning seeds")
    if args.cap < 0 or args.window < 1 or not 0 < args.rho <= 1:
        parser.error("Invalid cap or diagnostic window")
    output = args.output_dir or ROOT / "artifacts/results" / (
        "controlled_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(output_dir=str(output), cpt=asdict(SmoothCPT()), python=platform.python_version(),
                  numpy=np.__version__, started_at=datetime.now().isoformat(),
                  source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__).resolve(), ROOT / "mvp_cpt_pg/controlled_cpt.py"]})
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    start = time.perf_counter()
    if args.suite in ("all", "path"):
        path_experiments(output)
    if args.suite in ("all", "estimator"):
        estimator_experiments(args, output)
    if args.suite in ("all", "tracking"):
        tracking_experiments(args, output)
    (output / "completion.json").write_text(json.dumps(dict(
        completed_at=datetime.now().isoformat(), elapsed_seconds=time.perf_counter() - start,
        suite=args.suite, status="complete"), indent=2), encoding="utf-8")
    print("output", output, flush=True)


if __name__ == "__main__":
    main()
