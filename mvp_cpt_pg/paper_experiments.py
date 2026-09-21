"""Finite-horizon, multi-asset implementation of the optimized paper algorithm.

Learning and execution use the same inertial stochastic action map. All
learning trajectories restart at the frozen episode context.
"""

from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd
from scipy.special import digamma, logsumexp

from .controlled_cpt import quantile_weight
from .cpt_objective import CPTPreference


METHODS = ["dynamic", "symmetric", "static", "expected", "exponential", "equal_weight"]
ESTIMATORS = ["hybrid", "plugin"]


@dataclass(frozen=True)
class PaperConfig:
    horizon: int = 5
    gamma: float = .05
    a0: float = 1.0
    theta_radius: float = 2.0
    eta_gain: float = .2
    eta_loss: float = .05
    cost: float = .001
    n: int = 256
    evaluation_n: int = 2048
    evaluation_m: int = 1024
    window: int = 5
    rho: float = .9
    dimension: int = 10
    exponential_risk: float = 2.0
    trajectory_budget: int = 512
    trade_fraction: float = .2
    policy_sharpness: float = 1.0
    dirichlet_scale: float = 1.0
    smoothness_bound: float | None = None

    def __post_init__(self):
        if min(self.horizon, self.n, self.evaluation_n, self.evaluation_m, self.window) < 1:
            raise ValueError("Sample counts, horizon and diagnostic window must be positive")
        if not 2 <= self.dimension <= 10:
            raise ValueError("Require 2 <= dimension <= 10")
        if self.trajectory_budget < 2:
            raise ValueError("Need at least two complete training trajectories")
        if not 0 < self.trade_fraction <= 1:
            raise ValueError("Trade fraction must lie in (0,1]")
        if self.smoothness_bound is not None:
            if (not np.isfinite(self.smoothness_bound) or self.smoothness_bound <= 0 or
                    self.gamma * self.smoothness_bound > self.a0):
                raise ValueError("The certified step requires gamma * L <= a0")
        scalars = [self.gamma, self.a0, self.theta_radius, self.eta_gain, self.eta_loss,
                   self.cost, self.rho, self.trade_fraction, self.exponential_risk,
                   self.policy_sharpness]
        if not np.isfinite(scalars).all() or min(self.gamma, self.a0, self.theta_radius, self.exponential_risk) <= 0:
            raise ValueError("Invalid finite algorithm parameters")
        if not (0 <= self.eta_gain < 1 and 0 <= self.eta_loss < 1 and 0 <= self.cost < .25
                and 0 < self.rho <= 1):
            raise ValueError("Invalid adaptation, cost or window settings")
        if not np.isfinite(self.dirichlet_scale) or self.dirichlet_scale <= 0:
            raise ValueError("Dirichlet concentration scale must be positive")


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


def training_repetitions(episode, horizon=5):
    """Final-paper schedule for independent complete gradient outputs."""
    equivalent_episode = max(1, int(math.ceil(int(episode) * int(horizon) / 5.0)))
    return max(4, math.isqrt(equivalent_episode - 1) + 1)


def update_reference(wealth, reference, config):
    eta = np.where(wealth >= reference, config.eta_gain, config.eta_loss)
    return reference + eta * (wealth - reference)


def policy_features(features, previous, dimension):
    result = np.broadcast_to(features, (*previous.shape[:-1], *features.shape[-2:])).copy()
    # Only the actor changes: retain raw factors used by the market simulator.
    result[..., 8] *= result[..., 1]  # cycle x cash; a common offset cancels in mean weights
    result[..., -1] = previous
    return result[..., :dimension]


def policy(theta, features, rng=None, sharpness=1.0, concentration_scale=1.0):
    logits = sharpness * (features @ theta)
    # A common multiplier changes Dirichlet sampling variance, not the mean portfolio.
    alpha = concentration_scale * np.exp(logits)
    mean = alpha / alpha.sum(axis=-1, keepdims=True)
    if rng is None:
        return mean, None
    # Gamma(a+1)*U**(1/a) evaluated in log space avoids tiny-shape underflow.
    log_gamma = np.log(rng.gamma(alpha + 1)) - rng.exponential(size=alpha.shape) / alpha
    log_weight = log_gamma - logsumexp(log_gamma, axis=-1, keepdims=True)
    score = sharpness * np.einsum(
        "...i,...id->...d",
        alpha * (log_weight - digamma(alpha) +
                 digamma(alpha.sum(axis=-1, keepdims=True))),
        features,
    )
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


def action_map(previous, latent, tradable, trade_fraction):
    """Parameter-independent map used identically by training and execution.

    Turnover follows the manuscript's target-to-target L1 convention. This is
    not a holdings-drift or exact self-financing brokerage ledger.
    """
    feasible = executable_target(previous, latent, tradable)
    return (1 - trade_fraction) * previous + trade_fraction * feasible


def projected_update(theta, gradient, config):
    proposed = theta + config.gamma * gradient / max(np.linalg.norm(gradient), config.a0)
    norm = np.linalg.norm(proposed)
    return proposed * min(1.0, config.theta_radius / max(norm, np.finfo(float).tiny))


def objective(utilities, cpt):
    n = len(utilities)
    probabilities = 1 - np.arange(n + 1) / n
    values = np.sort(utilities, axis=0)
    intervals = np.diff(np.vstack([np.zeros(2), values]), axis=0)
    return float(intervals[:, 0] @ cpt.weight(probabilities[:-1], 0) -
                 intervals[:, 1] @ cpt.weight(probabilities[:-1], 1))


@dataclass
class EpisodeLaw:
    market: object
    day: int
    wealth: float
    reference: float
    previous: np.ndarray
    config: PaperConfig
    cpt: CPTPreference = CPTPreference()

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
                # Keep every bootstrapped state paired with its own historical
                # return and tradability mask. Mixing the current state with a
                # past return would destroy the factor-return relationship.
                features = self.market.panel.features[starts + offset]
            features = policy_features(features, previous, c.dimension)
            weights, score = policy(
                theta, features, rng, c.policy_sharpness, c.dirichlet_scale
            )
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
            weights = action_map(previous, weights, tradable, c.trade_fraction)
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


def _loo_components(utilities, scores, preference):
    """Return the raw CPT score and its two centered LOO controls.

    This is the same order-statistic decomposition used by the finalized E2
    estimator.  Keeping the decomposition here avoids routing E5 through the
    historical plug-in implementation.
    """
    u = np.asarray(utilities, float)
    g = np.asarray(scores, float)
    if g.ndim == 1:
        g = g[:, None]
    n = len(u)
    if n < 2 or u.shape != (n, 2) or g.shape[0] != n:
        raise ValueError("Expected utilities with shape (n, 2) and matching scores")
    if not np.isfinite(u).all() or not np.isfinite(g).all() or (u < 0).any():
        raise ValueError("LOO estimator requires finite nonnegative utilities")
    out = np.zeros((3, g.shape[1]))
    k = n - np.arange(n)
    p_yes = (k - 1) / (n - 1)
    p_no = k[1:] / (n - 1)
    for side, sign in ((0, 1.0), (1, -1.0)):
        order = np.argsort(u[:, side], kind="stable")
        gaps = np.diff(np.r_[0.0, u[order, side]])
        weights = preference.weight_prime(p_yes, side)
        raw = np.cumsum(gaps * weights)
        control = np.cumsum(gaps * weights * p_yes)
        upper = np.zeros(n)
        upper[1:] = gaps[1:] * preference.weight_prime(p_no, side) * p_no
        control += upper.sum() - np.cumsum(upper)
        out[0] += sign * np.mean(raw[:, None] * g[order], axis=0)
        out[1 + side] = sign * np.mean(control[:, None] * g[order], axis=0)
    return out


def _hybrid_design(budget):
    """Choose the E2 randomized-correction design under an expected budget."""
    budget = int(budget)
    decay = 2.0 ** -1.5
    kappa = 2.0 * (1.0 - decay) / (1.0 - 2.0 * decay)
    for n in range(budget, 1, -1):
        activation = n ** -0.5
        if n + activation * kappa * n <= budget:
            return n, activation, decay
    raise ValueError("trajectory_budget is too small for the hybrid estimator")


def hybrid(law, theta, rng, budget):
    """Centered LOO base plus an independent randomized debiasing correction."""
    n, activation, decay = _hybrid_design(budget)
    base_rng, control_rng, correction_rng = [
        np.random.default_rng(seed) for seed in rng.integers(0, 2**63 - 1, size=3)
    ]
    utilities, scores, _ = law.draw(theta, base_rng, n)
    parts = _loo_components(utilities, scores, law.cpt)
    estimate = parts[0] - parts[1:].sum(axis=0)
    if control_rng.random() >= activation:
        return estimate, n
    level = int(control_rng.geometric(1.0 - decay))
    correction_count = n * (1 << level)
    utilities, scores, _ = law.draw(theta, correction_rng, correction_count)
    half = correction_count // 2
    raw_full = _loo_components(utilities, scores, law.cpt)[0]
    raw_first = _loo_components(utilities[:half], scores[:half], law.cpt)[0]
    raw_second = _loo_components(utilities[half:], scores[half:], law.cpt)[0]
    probability = (1.0 - decay) * decay ** (level - 1)
    correction = (raw_full - 0.5 * (raw_first + raw_second)) / (activation * probability)
    return estimate + correction, n + correction_count


def estimate_gradient(law, theta, rng, method, estimator):
    c = law.config
    count = c.trajectory_budget
    if method in ("expected", "exponential"):
        _, score, terminal = law.draw(theta, rng, count)
        utility = terminal if method == "expected" else -np.exp(-c.exponential_risk * terminal)
        gradient = ((utility - utility.mean())[:, None] * score).sum(axis=0) / (count - 1)
        return gradient, count
    selected = "plugin" if method == "plugin" else estimator
    if selected == "hybrid":
        return hybrid(law, theta, rng, count)
    if selected == "plugin":
        if not 0 < c.n < count:
            raise ValueError("For plug-in, inner n must be strictly between 0 and total budget")
        return plugin(law, theta, rng, c.n, count - c.n)
    raise ValueError("Active estimator must be hybrid or plugin")


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


def run_online(market, config, seeds, methods, steps, start_day=0, estimator="hybrid",
               common_probe=False, average_training_outputs=False):
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
                        training_repeats = 0
                    else:
                        training_repeats = (training_repetitions(episode, c.horizon)
                                            if average_training_outputs else 1)
                        training_rng = rng_for(seed, episode, 10)
                        gradients = []
                        calls = 0
                        for _ in range(training_repeats):
                            estimate, estimate_calls = estimate_gradient(
                                law, theta, training_rng, method, estimator)
                            gradients.append(estimate)
                            calls += estimate_calls
                        gradient = np.mean(gradients, axis=0)
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
                    calls, training_repeats = 0, 0
                for offset in range(c.horizon):
                    index = day + offset
                    if method == "equal_weight":
                        target = np.r_[0.0, np.full(len(previous) - 1, 1 / (len(previous) - 1))]
                    else:
                        features = policy_features(market.panel.features[index], previous, c.dimension)
                        target, _ = policy(
                            theta, features, rng_for(seed, index, 40),
                            c.policy_sharpness, c.dirichlet_scale,
                        )
                    target = action_map(previous, target, market.panel.tradable[index], c.trade_fraction)
                    old_wealth = wealth
                    wealth, fee, turnover = next_wealth(wealth, previous, target, execution[index], c)
                    reference = float(update_reference(wealth, reference, c))
                    cumulative_fee += float(fee)
                    peak = max(peak, wealth)
                    risky = target[1:]
                    risky_total = float(risky.sum())
                    effective_holdings = (risky_total ** 2 / float(risky @ risky)
                                          if float(risky @ risky) > 0 else 0.0)
                    daily_rows.append(dict(seed=seed, method=method, episode=episode, day=index,
                        date=str(market.panel.dates[index]), regime=market.regime(index), wealth=float(wealth),
                        reference=reference, net_return=float(wealth / old_wealth - 1),
                        drawdown=float(1 - wealth / peak), cash=float(target[0]), turnover=float(turnover),
                        fee=float(fee), cumulative_fee=cumulative_fee,
                        effective_holdings=effective_holdings,
                        max_stock_weight=float(risky.max(initial=0.0))))
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
                    training_repeats=training_repeats, training_trajectories=calls,
                    training_steps=calls * c.horizon,
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
