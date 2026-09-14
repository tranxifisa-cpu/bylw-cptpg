"""Finite-horizon, multi-asset implementation of the optimized paper algorithm.

All learning trajectories restart at the saved episode context. No executed
mean-policy trajectory enters the stochastic gradient estimate.
"""

from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd
from scipy.special import digamma, logsumexp

from .controlled_cpt import (SmoothCPT, expected_cost, layer_probabilities,
                            paired_quantile_weight, quantile_weight)


METHODS = ["dynamic", "symmetric", "static", "expected", "exponential", "equal_weight"]
ESTIMATORS = ["plugin", "multilevel", "multilevel_outer", "split_base"]


@dataclass(frozen=True)
class PaperConfig:
    horizon: int = 5
    gamma: float = .05
    a0: float = 1.0
    theta_radius: float = 2.0
    eta_gain: float = .2
    eta_loss: float = .05
    cost: float = .001
    m: int = 64
    n: int = 256
    cap: int = 5
    base: int = 2
    exponent: float = 1.5
    evaluation_n: int = 2048
    evaluation_m: int = 1024
    window: int = 5
    rho: float = .9
    dimension: int = 10
    exponential_risk: float = 2.0
    outer_batch: int = 8
    trajectory_budget: int = 0

    def __post_init__(self):
        if min(self.horizon, self.m, self.n, self.evaluation_n, self.evaluation_m, self.window, self.base) < 1:
            raise ValueError("Sample counts, horizon and diagnostic window must be positive")
        if not 2 <= self.dimension <= 10 or not 0 <= self.cap <= 16:
            raise ValueError("Require 2 <= dimension <= 10 and 0 <= cap <= 16")
        if self.outer_batch < 1 or self.trajectory_budget < 0:
            raise ValueError("Invalid outer batch or trajectory budget")
        scalars = [self.gamma, self.a0, self.theta_radius, self.eta_gain, self.eta_loss,
                   self.cost, self.rho, self.exponent, self.exponential_risk]
        if not np.isfinite(scalars).all() or min(self.gamma, self.a0, self.theta_radius, self.exponential_risk) <= 0:
            raise ValueError("Invalid finite algorithm parameters")
        if not (0 <= self.eta_gain < 1 and 0 <= self.eta_loss < 1 and 0 <= self.cost < .25
                and 0 < self.rho <= 1 and 1 < self.exponent < 2):
            raise ValueError("Invalid adaptation, cost or layer settings")


def method_config(config, method):
    if method not in METHODS + ["plugin", "frozen"]:
        raise ValueError(method)
    if method == "static":
        return replace(config, eta_gain=0, eta_loss=0)
    if method == "symmetric":
        eta = (config.eta_gain + config.eta_loss) / 2
        return replace(config, eta_gain=eta, eta_loss=eta)
    return config


def rng_for(seed, episode, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(episode), int(stream)]))


def update_reference(wealth, reference, config):
    eta = np.where(wealth >= reference, config.eta_gain, config.eta_loss)
    return reference + eta * (wealth - reference)


def policy_features(features, previous, dimension):
    result = np.broadcast_to(features, (*previous.shape[:-1], *features.shape[-2:])).copy()
    result[..., -1] = previous
    return result[..., :dimension]


def policy(theta, features, rng=None):
    alpha = np.exp(features @ theta)
    if rng is None:
        return alpha / alpha.sum(axis=-1, keepdims=True), None
    # Gamma(a+1)*U**(1/a) evaluated in log space avoids tiny-shape underflow.
    log_gamma = np.log(rng.gamma(alpha + 1)) - rng.exponential(size=alpha.shape) / alpha
    log_weight = log_gamma - logsumexp(log_gamma, axis=-1, keepdims=True)
    score = np.einsum("...i,...id->...d", alpha * (log_weight - digamma(alpha) +
                     digamma(alpha.sum(axis=-1, keepdims=True))), features)
    return np.exp(log_weight), score


def next_wealth(wealth, previous, target, returns, config):
    turnover_l1 = abs(target - previous).sum(axis=-1)
    fee_fraction = config.cost * turnover_l1
    gross = 1 + (target * returns).sum(axis=-1) - fee_fraction
    if np.any(gross <= 0) or not np.isfinite(gross).all():
        raise FloatingPointError("Nonpositive wealth multiplier")
    return wealth * gross, wealth * fee_fraction, .5 * turnover_l1


def executable_target(previous, target, tradable):
    """Freeze unavailable assets and renormalize the remaining target."""
    previous = np.asarray(previous, dtype=float)
    target = np.asarray(target, dtype=float)
    tradable = np.asarray(tradable, dtype=bool)
    squeeze = previous.ndim == 1
    if squeeze:
        previous = previous[None, :]
        target = target[None, :]
        tradable = np.broadcast_to(tradable, previous.shape)
    result = np.zeros_like(target)
    frozen = ~tradable
    result[frozen] = previous[frozen]
    remaining = np.maximum(0.0, 1.0 - (result * frozen).sum(axis=1))
    flexible = np.where(tradable, np.maximum(target, 0.0), 0.0)
    total = flexible.sum(axis=1)
    result += np.divide(flexible * remaining[:, None], total[:, None],
                        out=np.zeros_like(flexible), where=total[:, None] > 0)
    no_flexible_target = total <= 0
    result[no_flexible_target, 0] += remaining[no_flexible_target]
    return result[0] if squeeze else result


def projected_update(theta, gradient, config):
    proposed = theta + config.gamma * gradient / max(np.linalg.norm(gradient), config.a0)
    norm = np.linalg.norm(proposed)
    return proposed * min(1.0, config.theta_radius / max(norm, np.finfo(float).tiny))


def objective(utilities, cpt):
    n = len(utilities)
    probabilities = 1 - np.arange(n + 1) / n
    values = np.sort(utilities, axis=0)
    intervals = np.diff(np.vstack([np.zeros(2), values]), axis=0)
    return float((intervals[:, 0] - intervals[:, 1]) @ cpt.weight(probabilities[:-1]))


@dataclass
class EpisodeLaw:
    market: object
    day: int
    wealth: float
    reference: float
    previous: np.ndarray
    config: PaperConfig
    cpt: SmoothCPT = SmoothCPT()

    def draw(self, theta, rng, count):
        c = self.config
        wealth = np.full(count, self.wealth)
        reference = np.full(count, self.reference)
        previous = np.broadcast_to(self.previous, (count, len(self.previous))).copy()
        scores = np.zeros((count, c.dimension))
        if not self.market.semi:
            # Frozen historical block simulator: every source date precedes this episode.
            if self.day <= c.horizon:
                raise ValueError("Real-data simulator needs a full historical block before the episode")
            starts = rng.integers(0, self.day - c.horizon, size=count)
        for offset in range(c.horizon):
            if self.market.semi:
                features = self.market.panel.features[self.day + offset]
            else:
                features = (self.market.panel.features[self.day] if offset == 0 else
                            self.market.panel.features[starts + offset])
            features = policy_features(features, previous, c.dimension)
            weights, score = policy(theta, features, rng)
            scores += score
            if self.market.semi:
                # Factors are a prescribed exogenous path; returns are newly sampled.
                full = np.broadcast_to(self.market.panel.features[self.day + offset],
                                       (count, *self.market.panel.features.shape[1:]))
                returns = self.market.draw_returns(self.day + offset, full, rng)
            else:
                returns = self.market.panel.returns[starts + offset]
            tradable = (np.broadcast_to(self.market.panel.tradable[self.day + offset],
                                        (count, previous.shape[1])) if self.market.semi else
                        self.market.panel.tradable[starts + offset])
            weights = executable_target(previous, weights, tradable)
            returns = np.where(tradable, returns, 0.0)
            wealth, _, _ = next_wealth(wealth, previous, weights, returns, c)
            reference = update_reference(wealth, reference, c)
            previous = weights
        return self.cpt.utilities(wealth - reference), scores, wealth


def plugin(law, theta, rng, n, m):
    inner, _, _ = law.draw(theta, rng, n)
    outer, scores, _ = law.draw(theta, rng, m)
    estimate = (quantile_weight(inner, outer, law.cpt)[:, None] * scores).mean(axis=0)
    return estimate, n + m


def multilevel(law, theta, rng, repeats):
    c = law.config
    p = layer_probabilities(c.cap, c.exponent)
    levels = rng.choice(len(p), size=repeats, p=p)
    estimates = np.empty((repeats, c.dimension))
    costs = 1 + c.base * 2 ** levels
    for level in range(c.cap + 1):
        ids = np.flatnonzero(levels == level)
        n = c.base * 2 ** level
        chunk = max(1, 8192 // (n * len(law.previous)))
        for start in range(0, len(ids), chunk):
            ix = ids[start:start + chunk]
            outer, scores, _ = law.draw(theta, rng, len(ix))
            inner, _, _ = law.draw(theta, rng, len(ix) * n)
            inner = inner.reshape(len(ix), n, 2)
            delta = paired_quantile_weight(inner, outer, law.cpt)
            if level:
                delta -= .5 * (paired_quantile_weight(inner[:, :n // 2], outer, law.cpt) +
                               paired_quantile_weight(inner[:, n // 2:], outer, law.cpt))
            estimates[ix] = scores * (delta / p[level])[:, None]
    return estimates, costs


def level_replications(law, theta, rng, level, repeats, outer_batch):
    """Independent inner batches; share all outer paths across each full/half pair."""
    c = law.config
    n = c.base * 2 ** level
    values = np.empty((repeats, c.dimension))
    chunk = max(1, 8192 // (max(n, outer_batch) * len(law.previous)))
    for start in range(0, repeats, chunk):
        count = min(chunk, repeats - start)
        inner, _, _ = law.draw(theta, rng, count * n)
        outer, scores, _ = law.draw(theta, rng, count * outer_batch)
        inner = inner.reshape(count, n, 2)
        outer = outer.reshape(count, outer_batch, 2)
        scores = scores.reshape(count, outer_batch, c.dimension)
        estimate = np.zeros((count, c.dimension))
        for j in range(outer_batch):
            delta = paired_quantile_weight(inner, outer[:, j], law.cpt)
            if level:
                delta -= .5 * (paired_quantile_weight(inner[:, :n // 2], outer[:, j], law.cpt)
                               + paired_quantile_weight(inner[:, n // 2:], outer[:, j], law.cpt))
            estimate += delta[:, None] * scores[:, j]
        values[start:start + count] = estimate / outer_batch
    return values, np.full(repeats, n + outer_batch, dtype=int)


def estimator_cost(config, estimator):
    """Expected full-trajectory cost of one replication, including the base term."""
    if estimator == "multilevel":
        return expected_cost(config.cap, config.base, config.exponent)
    p = layer_probabilities(config.cap, config.exponent)
    inner_cost = config.base * 2 ** np.arange(config.cap + 1)
    if estimator == "multilevel_outer":
        return float(p @ inner_cost + config.outer_batch)
    if estimator == "split_base":
        cost = config.base + config.outer_batch
        if config.cap:
            cost += float(p[1:] @ inner_cost[1:] / p[1:].sum() + config.outer_batch)
        return cost
    raise ValueError(estimator)


def improved_multilevel(law, theta, rng, repeats, estimator):
    c = law.config
    p = layer_probabilities(c.cap, c.exponent)
    if estimator == "split_base":
        values, costs = level_replications(law, theta, rng, 0, repeats, c.outer_batch)
        if c.cap == 0:
            return values, costs
        levels_available = np.arange(1, c.cap + 1)
        probability = p[1:] / p[1:].sum()
    elif estimator == "multilevel_outer":
        values = np.zeros((repeats, c.dimension))
        costs = np.zeros(repeats, dtype=int)
        levels_available = np.arange(c.cap + 1)
        probability = p
    else:
        raise ValueError(estimator)
    sampled = rng.choice(levels_available, size=repeats, p=probability)
    for level, chance in zip(levels_available, probability):
        ids = np.flatnonzero(sampled == level)
        if len(ids):
            delta, cost = level_replications(law, theta, rng, int(level), len(ids), c.outer_batch)
            values[ids] += delta / chance
            costs[ids] += cost
    return values, costs


def budgeted_gradient(law, theta, rng, estimator, budget):
    if estimator == "plugin":
        if budget <= law.config.n:
            raise ValueError("Trajectory budget must exceed plugin inner n")
        return plugin(law, theta, rng, law.config.n, budget - law.config.n)
    repeats = int(np.floor(budget / estimator_cost(law.config, estimator)))
    if repeats < 1:
        raise ValueError("Trajectory budget too small for one estimator replication")
    if estimator == "multilevel":
        values, costs = multilevel(law, theta, rng, repeats)
    else:
        values, costs = improved_multilevel(law, theta, rng, repeats, estimator)
    return values.mean(axis=0), int(costs.sum())


def estimate_gradient(law, theta, rng, method, estimator):
    c = law.config
    if method in ("expected", "exponential"):
        count = c.trajectory_budget or max(2, round(c.m * expected_cost(c.cap, c.base, c.exponent)))
        _, score, terminal = law.draw(theta, rng, count)
        utility = terminal if method == "expected" else -np.exp(-c.exponential_risk * terminal)
        # Leave-one-out centering does not introduce the sample-mean baseline bias.
        gradient = ((utility - utility.mean())[:, None] * score).sum(axis=0) / (count - 1)
        return gradient, count
    if c.trajectory_budget:
        return budgeted_gradient(law, theta, rng, "plugin" if method == "plugin" else estimator,
                                 c.trajectory_budget)
    if estimator == "plugin" or method == "plugin":
        return plugin(law, theta, rng, c.n, c.m)
    if estimator == "multilevel":
        values, costs = multilevel(law, theta, rng, c.m)
    else:
        values, costs = improved_multilevel(law, theta, rng, c.m, estimator)
    return values.mean(axis=0), int(costs.sum())


def evaluate(law, theta, rng, method):
    c = law.config
    inner, _, terminal = law.draw(theta, rng, c.evaluation_n)
    outer, scores, outer_terminal = law.draw(theta, rng, c.evaluation_m)
    if method in ("expected", "exponential"):
        utility = (outer_terminal if method == "expected" else
                   -np.exp(-c.exponential_risk * outer_terminal))
        g = ((utility - utility.mean())[:, None] * scores).sum(axis=0) / (len(utility) - 1)
        j = float(np.mean(terminal if method == "expected" else -np.exp(-c.exponential_risk * terminal)))
    else:
        g = (quantile_weight(inner, outer, law.cpt)[:, None] * scores).mean(axis=0)
        j = objective(inner, law.cpt)
    return j, g


def run_online(market, config, seeds, methods, steps, start_day=0, estimator="multilevel",
               common_probe=False):
    if start_day + steps > len(market.panel.dates) or steps % config.horizon:
        raise ValueError("Steps must fit the panel and be divisible by horizon")
    rows, daily_rows = [], []
    for seed in seeds:
        execution = market.execution_returns(seed)
        for method in methods:
            c = method_config(config, method)
            theta = np.zeros(c.dimension)
            wealth, reference = 1.0, 1.0
            previous = np.zeros(len(market.panel.codes))
            previous[0] = 1
            peak, cumulative_fee, dlr, cumulative_q = 1.0, 0.0, 0.0, 0.0
            residuals = []
            for episode, day in enumerate(range(start_day, start_day + steps, c.horizon), 1):
                law = EpisodeLaw(market, day, wealth, reference, previous.copy(), c)
                probe_record = {}
                if method != "equal_weight":
                    if method == "frozen":
                        gradient, calls = np.zeros_like(theta), 0
                    else:
                        gradient, calls = estimate_gradient(law, theta, rng_for(seed, episode, 10), method, estimator)
                    objective_value, diagnostic = evaluate(law, theta, rng_for(seed, episode, 20), method)
                    _, second = evaluate(law, theta, rng_for(seed, episode, 21), method)
                    q = (projected_update(theta, diagnostic, c) - theta) / c.gamma
                    cumulative_q += float(q @ q)
                    residuals.append(q)
                    recent = np.asarray(residuals[-c.window:])[::-1]
                    weights = c.rho ** np.arange(len(recent))
                    smooth = (recent * weights[:, None]).sum(axis=0) / weights.sum()
                    dlr += float(smooth @ smooth)
                    diagnostic_gap = float(np.linalg.norm(diagnostic - second))
                    if common_probe:
                        initial = np.zeros(len(previous))
                        initial[0] = 1
                        probe = EpisodeLaw(market, day, 1., 1., initial, config)
                        probe_j, probe_g1 = evaluate(probe, theta, rng_for(seed, episode, 30), "dynamic")
                        _, probe_g2 = evaluate(probe, theta, rng_for(seed, episode, 31), "dynamic")
                        probe_record = dict(common_objective=probe_j,
                            common_gradient_cross=float(probe_g1 @ probe_g2),
                            common_gradient_noise=float(np.sum((probe_g1 - probe_g2) ** 2) / 2))
                        for i in range(c.dimension):
                            probe_record[f"common_g1_{i}"] = probe_g1[i]
                            probe_record[f"common_g2_{i}"] = probe_g2[i]
                else:
                    gradient = np.zeros_like(theta)
                    objective_value, q, diagnostic_gap = np.nan, np.full_like(theta, np.nan), np.nan
                    calls = 0
                for offset in range(c.horizon):
                    index = day + offset
                    if method == "equal_weight":
                        target = np.r_[0.0, np.full(len(previous) - 1, 1 / (len(previous) - 1))]
                    else:
                        features = policy_features(market.panel.features[index], previous, c.dimension)
                        target, _ = policy(theta, features)
                    target = executable_target(previous, target, market.panel.tradable[index])
                    old_wealth = wealth
                    wealth, fee, turnover = next_wealth(wealth, previous, target, execution[index], c)
                    reference = float(update_reference(wealth, reference, c))
                    cumulative_fee += float(fee)
                    peak = max(peak, wealth)
                    daily_rows.append(dict(seed=seed, method=method, episode=episode, day=index,
                        date=str(market.panel.dates[index]), regime=market.regime(index), wealth=float(wealth),
                        reference=reference, net_return=float(wealth / old_wealth - 1),
                        drawdown=float(1 - wealth / peak), cash=float(target[0]), turnover=float(turnover),
                        fee=float(fee), cumulative_fee=cumulative_fee))
                    previous = target
                new_theta = theta if method in ("equal_weight", "frozen") else projected_update(theta, gradient, c)
                record = dict(seed=seed, method=method, episode=episode, day=day,
                    date=str(market.panel.dates[day]), regime=market.regime(day), wealth=float(wealth),
                    reference=reference, cash=float(previous[0]), objective=objective_value,
                    q_squared=float(q @ q), dynamic_local_regret=dlr if method != "equal_weight" else np.nan,
                    cumulative_q_squared=cumulative_q if method != "equal_weight" else np.nan,
                    average_q_squared=cumulative_q / episode if method != "equal_weight" else np.nan,
                    average_dynamic_local_regret=dlr / episode if method != "equal_weight" else np.nan,
                    diagnostic_gradient_gap=diagnostic_gap, gradient_squared=float(gradient @ gradient),
                    theta_norm=float(np.linalg.norm(theta)), update_norm=float(np.linalg.norm(new_theta - theta)),
                    training_trajectories=calls, training_steps=calls * c.horizon,
                    evaluation_trajectories=0 if method == "equal_weight" else
                        (4 if common_probe else 2) * (c.evaluation_n + c.evaluation_m))
                for i in range(c.dimension):
                    record[f"theta_{i}"] = theta[i]
                    record[f"gradient_{i}"] = gradient[i]
                    record[f"q_{i}"] = q[i]
                    record[f"diagnostic_g1_{i}"] = diagnostic[i] if method != "equal_weight" else np.nan
                    record[f"diagnostic_g2_{i}"] = second[i] if method != "equal_weight" else np.nan
                record.update(probe_record)
                rows.append(record)
                theta = new_theta
            print(f"Finished {method}, seed={seed}, episodes={steps // c.horizon}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def summarize(daily):
    rows = []
    for (method, seed), frame in daily.groupby(["method", "seed"], sort=False):
        returns = frame.net_return.to_numpy()
        sd = returns.std(ddof=1)
        turnover = frame.turnover.sum()
        rows.append(dict(method=method, seed=seed, terminal_wealth=frame.wealth.iloc[-1],
            annualized_return=frame.wealth.iloc[-1] ** (252 / len(frame)) - 1,
            sharpe=np.sqrt(252) * returns.mean() / sd if sd > 0 else np.nan,
            max_drawdown=frame.drawdown.max(), total_turnover=turnover,
            return_per_turnover=(frame.wealth.iloc[-1] - 1) / turnover if turnover > 0 else np.nan,
            cumulative_fee=frame.fee.sum(), mean_cash=frame.cash.mean()))
    return pd.DataFrame(rows)
