"""Exact LOO raw statistic and signed gain/loss zero-mean controls.

Uses the unchanged regularized TK preference and path-score implementation.
Returned rows are (raw, gain_control, signed_loss_control). The old centered
LOO is raw - gain_control - signed_loss_control. No gradient clipping.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear


def loo_components(utilities, scores, preference):
    u = np.asarray(utilities, float)
    g = np.asarray(scores, float)
    if g.ndim == 1:
        g = g[:, None]
    n = len(u)
    if n < 2 or u.shape != (n, 2) or g.ndim != 2 or len(g) != n:
        raise ValueError('Expected (n,2) utilities and (n,d) scores with n>=2')
    if not np.isfinite(u).all() or not np.isfinite(g).all() or (u < 0).any():
        raise ValueError('Finite, nonnegative utility magnitudes required')
    out = np.zeros((3, g.shape[1]))
    k = n - np.arange(n)
    p_yes = (k-1)/(n-1)
    p_no = k[1:]/(n-1)
    for side, sign in ((0, 1.), (1, -1.)):
        ix = np.argsort(u[:, side], kind='stable')
        gaps = np.diff(np.r_[0., u[ix, side]])
        wp = preference.weight_prime(p_yes, side)
        raw = np.cumsum(gaps * wp)
        control = np.cumsum(gaps * wp * p_yes)
        upper = np.zeros(n)
        upper[1:] = gaps[1:] * preference.weight_prime(p_no, side) * p_no
        control += upper.sum() - np.cumsum(upper)
        out[0] += sign * np.mean(raw[:, None] * g[ix], axis=0)
        out[1+side] = sign * np.mean(control[:, None] * g[ix], axis=0)
    return out


def fit_coefficients(raw, controls, bound=4., ridge=1e-3):
    """Independent-pilot covariance fit; ridge towards 1 and fixed bounds.

    controls: (replications, p, d). Minimize empirical covariance trace, not
    MSE to a numerical reference. Test data never enter this function.
    """
    r = np.asarray(raw, float)
    c = np.asarray(controls, float)
    if r.ndim != 2 or c.ndim != 3 or c.shape[0] != len(r) or c.shape[2] != r.shape[1]:
        raise ValueError('Invalid coefficient fitting shapes')
    rc = r-r.mean(axis=0)
    cc = c-c.mean(axis=0)
    p = c.shape[1]
    x = cc.transpose(0,2,1).reshape(-1,p)/np.sqrt(len(r)-1)
    y = rc.reshape(-1)/np.sqrt(len(r)-1)
    scale = float(np.sum(x*x)/p)
    if scale <= 1e-25:
        return np.ones(p), dict(fallback='degenerate_control', ridge_scale=scale)
    penalty = ridge*scale
    xx = np.vstack([x, np.sqrt(penalty)*np.eye(p)])
    yy = np.r_[y, np.sqrt(penalty)*np.ones(p)]
    res = lsq_linear(xx, yy, bounds=(-bound, bound), tol=1e-12)
    if not res.success:
        raise RuntimeError('Pilot coefficient fit failed: '+res.message)
    return res.x, dict(ridge_scale=penalty, condition_number=float(np.linalg.cond(x.T@x+penalty*np.eye(p))),
                      at_bound=bool(np.any(abs(res.x) >= bound-1e-6)))
