"""Controlled CPT oracle and estimators for the optimized manuscript.

This module does not alter the historical market runners. Each sample is one
cash/stock Beta-policy episode. The numerical oracle integrates the population
survival law independently of the sampled estimator.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq


@dataclass(frozen=True)
class SmoothCPT:
    epsilon: float = 0.05
    value_epsilon: float = 0.01
    alpha: float = 0.88
    loss_aversion: float = 2.25

    def __post_init__(self):
        if not (0 < self.epsilon < 0.5 and self.value_epsilon > 0):
            raise ValueError("Regularization must be strictly positive")
        if not (0 < self.alpha <= 1 and self.loss_aversion > 1):
            raise ValueError("Invalid value parameters")

    def weight(self, p):
        eps = self.epsilon
        return ((eps + (1 - 2 * eps) * np.asarray(p)) ** 3 - eps**3) / (
            (1 - eps) ** 3 - eps**3
        )

    def weight_prime(self, p):
        eps = self.epsilon
        return 3 * (1 - 2 * eps) * (eps + (1 - 2 * eps) * np.asarray(p)) ** 2 / (
            (1 - eps) ** 3 - eps**3
        )

    def magnitude(self, x):
        return (np.asarray(x) ** 2 + self.value_epsilon**2) ** (self.alpha / 2) - (
            self.value_epsilon**self.alpha
        )

    def inverse_magnitude(self, z):
        return np.sqrt(np.maximum(
            (np.asarray(z) + self.value_epsilon**self.alpha) ** (2 / self.alpha)
            - self.value_epsilon**2, 0.0
        ))

    def utilities(self, y):
        y = np.asarray(y)
        return np.stack((self.magnitude(np.maximum(y, 0)),
                         self.loss_aversion * self.magnitude(np.maximum(-y, 0))), axis=-1)


@dataclass(frozen=True)
class BetaMarket:
    probability: float = 0.6
    return_size: float = 0.04
    wealth: float = 1.0
    reference: float = 1.0
    eta_gain: float = 0.4
    eta_loss: float = 0.1
    cpt: SmoothCPT = SmoothCPT()

    def __post_init__(self):
        if not (0 <= self.probability <= 1 and 0 < self.return_size < 1):
            raise ValueError("Invalid market law")
        if not (self.wealth > 0 and self.reference >= 0):
            raise ValueError("Invalid episode starting context")
        if not (0 <= self.eta_gain < 1 and 0 <= self.eta_loss < 1):
            raise ValueError("This oracle requires adaptation rates below one")

    def draw(self, rng, theta, size):
        concentration = math.exp(theta)
        exponential = rng.exponential(size=size)
        stock_weight = np.exp(-exponential / concentration)
        returns = np.where(rng.random(size=size) < self.probability,
                           self.return_size, -self.return_size)
        terminal = self.wealth * (1 + stock_weight * returns)
        eta = np.where(terminal >= self.reference, self.eta_gain, self.eta_loss)
        relative = (1 - eta) * (terminal - self.reference)
        # This evaluates 1 + exp(theta) * log(weight) without log underflow.
        return self.cpt.utilities(relative), 1 - exponential

    def oracle(self, theta, order=128):
        """Return J and dJ/dtheta by population utility-survival quadrature."""
        theta = np.asarray(theta, dtype=float)
        flat = theta.reshape(-1)
        concentration = np.exp(flat)[:, None]
        objective = np.zeros(len(flat))
        gradient = np.zeros(len(flat))
        nodes, weights = quadrature_nodes(order)
        for sign, scale, eta in ((1, 1.0, self.eta_gain),
                                 (-1, self.cpt.loss_aversion, self.eta_loss)):
            extent = max((1 - eta) * (self.wealth * (1 + sign * self.return_size)
                                     - self.reference) * sign, 0.0)
            upper = float(scale * self.cpt.magnitude(extent))
            split = float(scale * self.cpt.magnitude(
                max((1 - eta) * (self.wealth - self.reference) * sign, 0.0)))
            edges = np.unique([0.0, min(split, upper), upper])
            for lo, hi in zip(edges[:-1], edges[1:]):
                z = lo + (hi - lo) * nodes
                level = self.reference + sign * self.cpt.inverse_magnitude(z / scale) / (1 - eta)
                survival = np.zeros((len(flat), order))
                derivative = np.zeros_like(survival)
                for ret, prob in ((self.return_size, self.probability),
                                  (-self.return_size, 1 - self.probability)):
                    t = np.clip((level / self.wealth - 1) / ret, 0, 1)
                    ta = t[None, :] ** concentration
                    log_t = np.log(t, out=np.zeros_like(t), where=t > 0)
                    dta = concentration * ta * log_t
                    greater = (sign == 1) == (ret > 0)
                    survival += prob * ((1 - ta) if greater else ta)
                    derivative += prob * ((-dta) if greater else dta)
                objective += sign * (hi - lo) * (self.cpt.weight(survival) @ weights)
                gradient += sign * (hi - lo) * (
                    (self.cpt.weight_prime(survival) * derivative) @ weights)
        if theta.ndim == 0:
            return float(objective[0]), float(gradient[0])
        return objective.reshape(theta.shape), gradient.reshape(theta.shape)


@lru_cache(maxsize=12)
def quadrature_nodes(order):
    nodes, weights = leggauss(order)
    return (nodes + 1) / 2, weights / 2


def quantile_weight(inner, targets, cpt=SmoothCPT()):
    """Centered CPT weight, using independent utility order statistics."""
    inner = np.asarray(inner, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if inner.ndim != 2 or inner.shape[1] != 2 or len(inner) == 0:
        raise ValueError("Inner utilities must have shape (n, 2), n > 0")
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("Outer utilities must have shape (m, 2)")
    if not np.isfinite(inner).all() or not np.isfinite(targets).all():
        raise ValueError("Utilities must be finite")
    if np.any(inner < 0) or np.any(targets < 0):
        raise ValueError("Utility magnitudes must be nonnegative")
    result = np.zeros(len(targets))
    p = 1 - np.arange(len(inner) + 1) / len(inner)
    dw = cpt.weight_prime(p)
    for column, sign in ((0, 1), (1, -1)):
        q = np.sort(inner[:, column])
        edges = np.r_[0.0, q]
        segments = np.diff(edges) * dw[:-1]
        prefix = np.r_[0.0, np.cumsum(segments)]
        k = np.searchsorted(q, targets[:, column], side="right")
        integral = prefix[k] + (targets[:, column] - edges[k]) * dw[k]
        result += sign * (integral - np.dot(segments, p[:-1]))
    return result


def paired_quantile_weight(inner, outer, cpt):
    """Vectorized independent inner batches, one outer utility per batch."""
    count, n, _ = inner.shape
    p = 1 - np.arange(n + 1) / n
    dw = cpt.weight_prime(p)
    result = np.zeros(count)
    row = np.arange(count)
    for column, sign in ((0, 1), (1, -1)):
        q = np.sort(inner[:, :, column], axis=1)
        edges = np.c_[np.zeros(count), q]
        segments = np.diff(edges, axis=1) * dw[:-1]
        prefix = np.c_[np.zeros(count), np.cumsum(segments, axis=1)]
        target = outer[:, column]
        k = np.sum(q <= target[:, None], axis=1)
        integral = prefix[row, k] + (target - edges[row, k]) * dw[k]
        result += sign * (integral - np.sum(segments * p[:-1], axis=1))
    return result


def layer_probabilities(cap=7, exponent=1.5):
    if not isinstance(cap, int) or cap < 0 or not 1 < exponent < 2:
        raise ValueError("Require a nonnegative integer cap and 1 < exponent < 2")
    p = 2.0 ** (-exponent * np.arange(cap + 1))
    return p / p.sum()


def expected_cost(cap=7, base=2, exponent=1.5):
    if base < 1:
        raise ValueError("Base sample count must be positive")
    p = layer_probabilities(cap, exponent)
    return float(np.dot(p, 1 + base * 2 ** np.arange(cap + 1)))


def capped_estimates(market, theta, rng, repeats, cap=7, base=2, exponent=1.5):
    """IID capped single-term estimators, including all inner/outer costs."""
    if repeats < 1 or base < 1:
        raise ValueError("Sample counts must be positive")
    p = layer_probabilities(cap, exponent)
    levels = rng.choice(cap + 1, size=repeats, p=p)
    values = np.empty(repeats)
    costs = 1 + base * 2**levels
    for ell in range(cap + 1):
        positions = np.flatnonzero(levels == ell)
        n = base * 2**ell
        chunk = max(1, min(1024, 262144 // n))
        for start in range(0, len(positions), chunk):
            ids = positions[start:start + chunk]
            outer, score = market.draw(rng, theta, len(ids))
            inner, _ = market.draw(rng, theta, (len(ids), n))
            fine = paired_quantile_weight(inner, outer, market.cpt)
            if ell:
                first = paired_quantile_weight(inner[:, :n // 2], outer, market.cpt)
                second = paired_quantile_weight(inner[:, n // 2:], outer, market.cpt)
                fine -= (first + second) / 2
            values[ids] = fine * score / p[ell]
    return values, costs


def plugin_estimate(market, theta, rng, n, m):
    if min(n, m) < 1:
        raise ValueError("Both sample groups must be positive")
    inner, _ = market.draw(rng, theta, n)
    outer, scores = market.draw(rng, theta, m)
    return float(np.mean(quantile_weight(inner, outer, market.cpt) * scores))


def projected_update(theta, gradient, gamma, a0=1.0, bounds=(-2.0, 2.0)):
    if (not np.isfinite([theta, gradient, gamma, a0, *bounds]).all()
            or gamma <= 0 or a0 <= 0 or bounds[0] >= bounds[1]):
        raise ValueError("Invalid update configuration")
    return float(np.clip(theta + gamma * gradient / max(abs(gradient), a0), *bounds))


def residual(theta, gradient, gamma, a0=1.0, bounds=(-2.0, 2.0)):
    return (projected_update(theta, gradient, gamma, a0, bounds) - theta) / gamma


def stationary_points(market, bounds=(-2.0, 2.0), points=129, order=256):
    """Numerical grid/bracket roots and constrained endpoints, not a certificate."""
    grid = np.linspace(*bounds, points)
    _, gradient = market.oracle(grid, order)
    roots = []
    if gradient[0] <= 0:
        roots.append(bounds[0])
    if gradient[-1] >= 0:
        roots.append(bounds[1])
    for i in range(points - 1):
        if gradient[i] == 0:
            roots.append(float(grid[i]))
        elif gradient[i] * gradient[i + 1] < 0:
            roots.append(brentq(lambda t: market.oracle(t, order)[1], grid[i], grid[i + 1]))
    if not roots:
        raise RuntimeError("No numerical stationary point found")
    return np.unique(roots)


def residual_metrics(values, window=5, rho=0.9):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or window < 1 or not 0 < rho <= 1:
        raise ValueError("Invalid scalar residual sequence")
    smooth = np.empty(len(values))
    for k in range(len(values)):
        recent = values[max(0, k - window + 1):k + 1][::-1]
        weights = rho ** np.arange(len(recent))
        smooth[k] = np.dot(weights, recent) / weights.sum()
    cumulative = np.cumsum(smooth**2)
    average_q = np.cumsum(values**2) / np.arange(1, len(values) + 1)
    return smooth, cumulative, cumulative / np.arange(1, len(values) + 1), average_q
