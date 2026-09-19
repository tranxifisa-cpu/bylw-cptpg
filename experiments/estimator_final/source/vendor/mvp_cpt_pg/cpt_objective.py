"""The single preference specification used by the revised paper and runner.

Probability weighting is endpoint-normalized TK on a fixed interior interval.
Value magnitudes use shifted powers, preserving gain concavity/loss convexity.
No fit or target changes with the gradient batch size.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CPTPreference:
    epsilon: float = .05
    value_epsilon: float = .01
    alpha: float = .88
    loss_aversion: float = 2.25
    beta_gain: float = .61
    beta_loss: float = .69

    def __post_init__(self):
        parameters = tuple(self.__dict__.values())
        if not np.isfinite(parameters).all():
            raise ValueError("Preference parameters must be finite")
        if not (0 < self.epsilon < .5 and self.value_epsilon > 0):
            raise ValueError("Regularization must be fixed and strictly positive")
        if not (0 < self.alpha <= 1 and self.loss_aversion > 1):
            raise ValueError("Require 0 < alpha <= 1 and loss aversion > 1")
        if not all(.5 <= b <= 1 for b in (self.beta_gain, self.beta_loss)):
            raise ValueError("This release uses the proven monotone TK range [0.5, 1]")

    @staticmethod
    def _tk(q, beta):
        return q ** beta / (q ** beta + (1 - q) ** beta) ** (1 / beta)

    def _arguments(self, p, side):
        p = np.asarray(p, dtype=float)
        if side not in (0, 1) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Side is 0/1 and probabilities must lie in [0,1]")
        beta = (self.beta_gain, self.beta_loss)[side]
        lo = self._tk(self.epsilon, beta)
        denominator = self._tk(1 - self.epsilon, beta) - lo
        return self.epsilon + (1 - 2 * self.epsilon) * p, beta, lo, denominator

    def weight(self, p, side=0):
        q, beta, lo, denominator = self._arguments(p, side)
        return (self._tk(q, beta) - lo) / denominator

    def weight_prime(self, p, side=0):
        q, beta, _, denominator = self._arguments(p, side)
        total = q ** beta + (1 - q) ** beta
        log_derivative = beta / q - (q ** (beta - 1) - (1 - q) ** (beta - 1)) / total
        return (1 - 2 * self.epsilon) * self._tk(q, beta) * log_derivative / denominator

    def magnitude(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0):
            raise ValueError("Value magnitudes require nonnegative arguments")
        return (x + self.value_epsilon) ** self.alpha - self.value_epsilon ** self.alpha

    def inverse_magnitude(self, z):
        return np.maximum((np.asarray(z) + self.value_epsilon ** self.alpha) **
                          (1 / self.alpha) - self.value_epsilon, 0.)

    def utilities(self, relative):
        relative = np.asarray(relative, dtype=float)
        return np.stack((self.magnitude(np.maximum(relative, 0)),
                         self.loss_aversion * self.magnitude(np.maximum(-relative, 0))), axis=-1)


def pooled_gradient(utilities, scores, preference):
    """Exact sorted evaluation of the leave-one-out statistic in the paper.

    All B paths are independent draws at one frozen context. Each outer path
    uses the OTHER B-1 paths for its empirical survivals. The resulting B terms
    are dependent; they must not be treated as iid replications for an SE.
    Finite-sample bias is generally O(1/B), not zero. Cost is exactly B paths.
    """
    utilities, scores = np.asarray(utilities, float), np.asarray(scores, float)
    scalar = scores.ndim == 1
    if scalar:
        scores = scores[:, None]
    count = len(utilities)
    if (count < 2 or utilities.shape != (count, 2) or scores.ndim != 2 or
            len(scores) != count or np.any(utilities < 0) or
            not np.isfinite(utilities).all() or not np.isfinite(scores).all()):
        raise ValueError("Need B >= 2 finite path utilities (B,2) and scores (B,d)")
    n = count - 1
    k = count - np.arange(count)
    positive = (k - 1) / n
    negative = k[1:] / n
    result = np.zeros(scores.shape[1])
    for side, sign in ((0, 1), (1, -1)):
        order = np.argsort(utilities[:, side], kind="stable")
        gaps = np.diff(np.r_[0., utilities[order, side]])
        below = np.cumsum(gaps * preference.weight_prime(positive, side) * (1 - positive))
        above = np.zeros(count)
        above[1:] = gaps[1:] * preference.weight_prime(negative, side) * negative
        coefficient = below - (above.sum() - np.cumsum(above))
        result += sign * np.mean(coefficient[:, None] * scores[order], axis=0)
    return float(result[0]) if scalar else result
