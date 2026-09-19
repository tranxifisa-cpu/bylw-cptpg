"""Formula-level v5 baselines and the current regularized TK objective.

These are not runs of the reference authors' repository. All estimators target
the same EpisodeLaw and preferences. Independent split blocks have no overlap.
"""
import numpy as np
from mvp_cpt_pg.controlled_cpt import quantile_weight
from mvp_cpt_pg.cpt_objective import pooled_gradient


def v5_weight(inner, outer, cpt):
    n = len(inner)
    survival = 1 - np.arange(n + 1) / n
    result = np.zeros(len(outer))
    for side, sign in ((0, 1), (1, -1)):
        q = np.sort(inner[:, side])
        edges = np.r_[0., q]
        wp = cpt.weight_prime(survival, side)
        prefix = np.r_[0., np.cumsum(np.diff(edges) * wp[:-1])]
        k = np.searchsorted(q, outer[:, side], side='right')
        result += sign * (prefix[k] + (outer[:, side] - edges[k]) * wp[k])
    return result


def split_estimate(utilities, scores, cpt, fraction, centered=False):
    n = int(len(utilities) * fraction)
    if not 1 <= n < len(utilities):
        raise ValueError('Both independent blocks must be nonempty')
    weight = quantile_weight if centered else v5_weight
    return np.mean(weight(utilities[:n], utilities[n:], cpt)[:, None] * scores[n:], axis=0)


def chunked_draw(law, theta, rng, count, chunk=8192):
    """Bound temporary path-state memory without capping the random level.

    Every requested path is completed. Chunking only changes the pseudorandom
    stream order; it does not truncate, clip, resample or reject a path.
    """
    u = np.empty((count, 2))
    s = np.empty((count, len(theta)))
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        u[start:stop], s[start:stop], _ = law.draw(theta, rng, stop - start)
    return u, s
