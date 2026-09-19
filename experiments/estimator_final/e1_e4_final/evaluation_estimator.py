"""Independent high-precision unbiased evaluator for E3.

Training and evaluation are statistically separate.  Each complete evaluation
replicate is an independent split estimator with an independent randomized
correction.  Averaging complete replicates reduces evaluation variance without
changing the target gradient.
"""
from __future__ import annotations

import numpy as np

from estimators import chunked_draw
from split_debias import design_for_budget, estimate
from mvp_cpt_pg.controlled_cpt import quantile_weight


def evaluate_gradient(law, theta, seed, *, budget: int = 1536, repeats: int = 2,
                      inner_fraction: float = 2/3):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    design = design_for_budget(budget, inner_fraction=inner_fraction, exponent=1.5)
    root = np.random.SeedSequence(seed)
    values = []
    actual = 0
    levels = []
    for child in root.spawn(repeats):
        rng = np.random.default_rng(child)

        def draw(random, count):
            return chunked_draw(law, theta, random, count)

        def statistic(inner, outer, scores):
            return np.mean(quantile_weight(inner, outer, law.cpt)[:, None] * scores, axis=0)

        value, info = estimate(draw, statistic, rng, design)
        values.append(np.asarray(value, dtype=float))
        actual += int(info['calls'])
        levels.append(int(info['level']))
    values = np.asarray(values)
    return values.mean(axis=0), dict(
        repeats=int(repeats),
        nominal_budget_per_repeat=int(budget),
        expected_paths=float(repeats * design.expected_cost),
        actual_paths=int(actual),
        maximum_level=int(max(levels)),
        replicate_gradients=values,
        design=dict(n=design.n, m=design.m, activation=design.rho,
                    exponent=design.exponent, expected_cost=design.expected_cost),
    )
