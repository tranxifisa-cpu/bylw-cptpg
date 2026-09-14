"""Fixed-context, cost-matched estimator comparisons and level diagnostics."""

from dataclasses import replace
import time

import numpy as np
import pandas as pd

from .paper_experiments import (EpisodeLaw, budgeted_gradient, evaluate,
    level_replications, rng_for)


def run_diagnostics(args, market, config):
    if min(args.replications, args.reference_batches) < 2:
        raise ValueError("Diagnostics require at least two repetitions and reference batches")
    if config.n != config.base * 2 ** config.cap:
        raise ValueError("Use n = base * 2**cap to compare identical finite-level expectations")
    if min(args.budget_grid) <= config.n or len(set(args.budget_grid)) != len(args.budget_grid):
        raise ValueError("Distinct budgets must each exceed n")
    if len(set(args.diagnostic_days)) != len(args.diagnostic_days):
        raise ValueError("Diagnostic days must be distinct")
    if any(day < 0 or day + config.horizon > len(market.panel.dates) for day in args.diagnostic_days):
        raise ValueError("Diagnostic horizon does not fit the panel")
    theta = np.full(config.dimension, .1 / np.sqrt(config.dimension))
    previous = np.zeros(len(market.panel.codes))
    previous[0] = 1
    raw, summaries, level_raw, level_summary, references = [], [], [], [], []
    for day in args.diagnostic_days:
        law = EpisodeLaw(market, day, 1., 1., previous.copy(), config)
        for precision, scale in [("half", 1), ("full", 2)]:
            reference_law = replace(law, config=replace(config,
                evaluation_n=args.reference_n * scale, evaluation_m=args.reference_m * scale))
            for batch in range(args.reference_batches):
                j, g = evaluate(reference_law, theta, rng_for(71000 + day, batch, scale), "dynamic")
                references.append(dict(day=day, regime=market.regime(day), precision=precision,
                    batch=batch, objective=j, **{f"g{i}": v for i, v in enumerate(g)}))
        ref_table = pd.DataFrame(references)
        cols = [f"g{i}" for i in range(config.dimension)]
        ref_values = ref_table[(ref_table.day == day) & (ref_table.precision == "full")][cols]
        truth = ref_values.mean().to_numpy()
        ref_var = ref_values.var(ddof=1).to_numpy() / len(ref_values)
        pd.DataFrame(references).to_csv(args.output / "diagnostic_references.csv", index=False)
        for outer_batch in sorted({1, config.outer_batch}):
            for level in range(config.cap + 1):
                values, costs = level_replications(law, theta,
                    rng_for(72000 + day, level, outer_batch), level, args.replications, outer_batch)
                level_summary.append(dict(day=day, regime=market.regime(day), level=level,
                    outer_batch=outer_batch, cost=float(costs.mean()),
                    second_moment=float(np.mean(np.sum(values ** 2, axis=1))),
                    variance_trace=float(values.var(axis=0, ddof=1).sum()),
                    mean_norm=float(np.linalg.norm(values.mean(axis=0)))))
                for replication, (g, cost) in enumerate(zip(values, costs)):
                    level_raw.append(dict(day=day, level=level, outer_batch=outer_batch,
                        replication=replication, cost=int(cost), **{f"g{i}": v for i, v in enumerate(g)}))
        for budget in args.budget_grid:
            for index, estimator in enumerate(args.compare_estimators):
                values, costs, seconds = [], [], []
                for replication in range(args.replications):
                    started = time.perf_counter()
                    g, cost = budgeted_gradient(law, theta,
                        rng_for(73000 + day, replication, 10 * budget + index), estimator, budget)
                    elapsed = time.perf_counter() - started
                    values.append(g)
                    costs.append(cost)
                    seconds.append(elapsed)
                    raw.append(dict(day=day, regime=market.regime(day), estimator=estimator,
                        budget=budget, replication=replication, cost=cost, seconds=elapsed,
                        **{f"g{i}": v for i, v in enumerate(g)}))
                values = np.asarray(values)
                errors = np.sum((values - truth) ** 2, axis=1)
                summaries.append(dict(day=day, regime=market.regime(day), estimator=estimator,
                    budget=budget, cost=np.mean(costs), cost_std=np.std(costs, ddof=1),
                    seconds=np.mean(seconds), mse=errors.mean(),
                    mse_se=errors.std(ddof=1) / np.sqrt(len(errors)),
                    variance_trace=values.var(axis=0, ddof=1).sum(),
                    bias_norm=np.linalg.norm(values.mean(axis=0) - truth),
                    bias_coordinate=values[:, 0].mean() - truth[0],
                    bias_se=np.sqrt(values[:, 0].var(ddof=1) / len(values) + ref_var[0]),
                    reference_variance_trace=ref_var.sum(), reference_gradient_norm=np.linalg.norm(truth)))
                for name, rows in [("diagnostic_raw", raw), ("diagnostic_summary", summaries),
                                   ("level_raw", level_raw), ("level_summary", level_summary)]:
                    pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)
                print(f"Diagnostic day={day}, budget={budget}, estimator={estimator}: "
                      f"MSE={errors.mean():.6g}, mean cost={np.mean(costs):.1f}", flush=True)
