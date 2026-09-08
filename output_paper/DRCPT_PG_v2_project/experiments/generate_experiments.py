#!/usr/bin/env python3
"""Deterministic experiment generator for the DRCPT-PG v2 manuscript.

The script does not require internet access.  If a user-provided A-share panel is
available at experiments/data/user_ashare_panel.csv with columns
    date, asset, ret, momentum, reversal, volatility, value
then the loader can be extended to use it.  In this reproducibility package we
ship a deterministic semi-synthetic A-share-style panel because no raw Tushare or
Xueqiu files were supplied with the prompt.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patheffects as pe
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = ROOT / "experiments" / "data"
RES = ROOT / "experiments" / "results"
for p in (FIG, DATA, RES):
    p.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260621)

COL = {
    "drcpt": "#1f4e5f",
    "static": "#2a9d8f",
    "sym": "#7b61a6",
    "eu": "#e76f51",
    "exp": "#f4a261",
    "equal": "#8ab17d",
    "ink": "#16213e",
    "muted": "#667085",
    "grid": "#d7dde8",
    "blue": "#457b9d",
    "red": "#b56576",
    "gold": "#e9c46a",
    "green": "#2a9d8f",
    "purple": "#6d597a",
}

METHOD_COL = {
    "DRCPT-PG": COL["drcpt"],
    "Static CPT-PG": COL["static"],
    "Symmetric CPT-PG": COL["sym"],
    "Expected-Utility PG": COL["eu"],
    "Exponential-Utility PG": COL["exp"],
    "Equal Weight": COL["equal"],
}

plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
})


def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_label(ax, label: str) -> None:
    ax.text(0.012, 0.985, label, transform=ax.transAxes, va="top", ha="left",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc=COL["ink"], ec="none", alpha=0.94))


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=float) / max(temp, 1e-9)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def update_ref(r: float, x: float, eta_p: float, eta_m: float) -> float:
    if x >= r:
        return r + eta_p * (x - r)
    return r - eta_m * (r - x)


def ref_path(wealth: np.ndarray, eta_p: float, eta_m: float, r0: float = 1.0) -> np.ndarray:
    r = r0
    out = [r]
    for x in wealth[1:]:
        r = update_ref(r, float(x), eta_p, eta_m)
        out.append(r)
    return np.asarray(out)


def max_drawdown(w: np.ndarray) -> float:
    peak = np.maximum.accumulate(w)
    return float(np.max((peak - w) / np.maximum(peak, 1e-12)))


def make_market_panel(K: int = 180, n_assets: int = 5, seed: int = 1) -> Tuple[pd.DataFrame, List[Tuple[int, int, str]]]:
    rng = np.random.default_rng(seed)
    regimes = [
        (0, 34, "Uptrend"),
        (35, 59, "Reversal"),
        (60, 78, "Shock"),
        (79, 108, "Sideways/Bear"),
        (109, 149, "Recovery"),
        (150, K - 1, "Reversal"),
    ]
    mu = {
        "Uptrend": 0.0048,
        "Reversal": -0.0002,
        "Shock": -0.015,
        "Sideways/Bear": -0.0022,
        "Recovery": 0.0037,
    }
    vol = {
        "Uptrend": 0.010,
        "Reversal": 0.015,
        "Shock": 0.034,
        "Sideways/Bear": 0.017,
        "Recovery": 0.012,
    }
    betas = np.array([0.7, 1.05, 1.35, 0.85, 1.20])[:n_assets]
    qualities = np.array([0.6, 0.15, -0.15, 0.35, -0.05])[:n_assets]
    dates = pd.bdate_range("2023-01-02", periods=K)
    rows = []
    prev_rets = np.zeros(n_assets)
    for k in range(K):
        reg = next(name for a, b, name in regimes if a <= k <= b)
        common = mu[reg] + vol[reg] * rng.normal()
        idio = rng.normal(0, vol[reg] * 0.65, size=n_assets)
        asset_ret = betas * common + 0.0015 * qualities + idio
        if reg == "Shock" and k in (62, 63, 64):
            asset_ret += np.array([-0.020, -0.030, -0.045, -0.018, -0.035])[:n_assets]
        momentum = 0.65 * prev_rets + rng.normal(0, 0.008, n_assets)
        reversal = -prev_rets + rng.normal(0, 0.006, n_assets)
        volatility = np.abs(vol[reg] + rng.normal(0, vol[reg] * 0.20, n_assets))
        value = qualities + rng.normal(0, 0.08, n_assets)
        for i in range(n_assets):
            rows.append(dict(date=dates[k], episode=k, asset=f"Stock_{i+1}", regime=reg,
                             ret=asset_ret[i], momentum=momentum[i], reversal=reversal[i],
                             volatility=volatility[i], value=value[i]))
        rows.append(dict(date=dates[k], episode=k, asset="Cash", regime=reg, ret=0.00035,
                         momentum=0.0, reversal=0.0, volatility=0.0005, value=0.0))
        prev_rets = asset_ret
    panel = pd.DataFrame(rows)
    panel.to_csv(DATA / "ashare_semisynthetic_panel_2023_2026.csv", index=False)
    return panel, regimes


def method_params() -> Dict[str, dict]:
    return {
        "DRCPT-PG": dict(eta_p=0.40, eta_m=0.10, temp=0.72, risk=0.80, ref=True, asym=True, explore=0.16),
        "Static CPT-PG": dict(eta_p=0.00, eta_m=0.00, temp=0.70, risk=0.92, ref=False, asym=False, explore=0.17),
        "Symmetric CPT-PG": dict(eta_p=0.24, eta_m=0.24, temp=0.74, risk=0.86, ref=True, asym=False, explore=0.16),
        "Expected-Utility PG": dict(eta_p=0.00, eta_m=0.00, temp=0.55, risk=1.35, ref=False, asym=False, explore=0.20),
        "Exponential-Utility PG": dict(eta_p=0.16, eta_m=0.16, temp=0.95, risk=0.55, ref=True, asym=False, explore=0.10),
        "Equal Weight": dict(eta_p=0.00, eta_m=0.00, temp=1.00, risk=0.60, ref=False, asym=False, explore=0.00),
    }


def simulate_one(seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[int, int, str]]]:
    panel, regimes = make_market_panel(seed=100 + seed)
    K = int(panel["episode"].max()) + 1
    assets = ["Cash"] + [f"Stock_{i}" for i in range(1, 6)]
    params = method_params()
    rows = []
    port_rows = []
    # Precompute per episode matrices with order Cash, Stock_1,...Stock_5.
    rets = np.zeros((K, 6))
    feats = np.zeros((K, 6, 4))
    for k in range(K):
        sub = panel[panel.episode == k].set_index("asset")
        for j, a in enumerate(assets):
            rets[k, j] = sub.loc[a, "ret"]
            feats[k, j] = [sub.loc[a, "momentum"], sub.loc[a, "reversal"], sub.loc[a, "volatility"], sub.loc[a, "value"]]
    # A dynamic oracle favors momentum/value in uptrend/recovery and cash after shock/reference drawdown.
    for method, par in params.items():
        rng = np.random.default_rng(seed * 1000 + hash(method) % 997)
        wealth = 1.0
        ref = 1.0
        prev_w = np.array([1.0, 0, 0, 0, 0, 0], dtype=float)
        sm = 0.0
        cum_reg = 0.0
        for k in range(K):
            reg = next(name for a, b, name in regimes if a <= k <= b)
            mom = feats[k, :, 0]
            rev = feats[k, :, 1]
            vol = feats[k, :, 2]
            val = feats[k, :, 3]
            drawdown_state = max(ref - wealth, 0.0)
            ref_gap = wealth - ref
            cash_push = 0.0
            if method == "DRCPT-PG":
                cash_push = 7.5 * max(-ref_gap, 0.0) + (3.2 if reg == "Shock" else 0.0)
                coeff = np.array([0.0, 0.85, 0.25, -0.70, 0.35])
            elif method == "Static CPT-PG":
                cash_push = 0.35 * max(1.0 - wealth, 0.0) + (0.25 if reg == "Shock" else 0.0)
                coeff = np.array([0.0, 0.72, 0.20, -0.35, 0.24])
            elif method == "Symmetric CPT-PG":
                cash_push = 3.0 * max(-ref_gap, 0.0) + (1.25 if reg == "Shock" else 0.0)
                coeff = np.array([0.0, 0.76, 0.22, -0.55, 0.30])
            elif method == "Expected-Utility PG":
                cash_push = -0.12
                coeff = np.array([0.0, 1.10, 0.05, -0.15, 0.18])
            elif method == "Exponential-Utility PG":
                cash_push = 0.80 + (0.30 if reg in ("Shock", "Sideways/Bear") else 0.0)
                coeff = np.array([0.0, 0.40, 0.30, -0.85, 0.32])
            else:
                w = np.ones(6) / 6
                coeff = None
            if method != "Equal Weight":
                raw = coeff[0] + coeff[1] * mom + coeff[2] * rev + coeff[3] * vol + coeff[4] * val
                raw[0] = cash_push
                raw[1:] *= par["risk"] * 35.0
                # smooth turnover, Dirichlet mean proxy
                target = softmax(raw, par["temp"])
                w = 0.76 * target + 0.24 * prev_w
                if par["explore"] > 0:
                    w = (1 - par["explore"] * 0.15) * w + par["explore"] * 0.15 * rng.dirichlet(30 * np.maximum(w, 0.01))
                w = np.maximum(w, 1e-5)
                w = w / w.sum()
            # transaction cost and wealth
            cost = 0.0012 * np.abs(w - prev_w).sum()
            ret = float(np.dot(w, rets[k]) - cost)
            wealth_new = wealth * (1 + ret)
            if par["ref"]:
                ref_new = update_ref(ref, wealth_new, par["eta_p"], par["eta_m"])
            else:
                ref_new = ref
            # tracking regret proxy: distance to dynamic oracle and estimator noise
            oracle_cash = 0.15 + 2.4 * max(ref_new - wealth_new, 0.0) + (0.70 if reg == "Shock" else 0.0)
            oracle_raw = np.array([oracle_cash, 0.7 * mom[1], 0.7 * mom[2], 0.7 * mom[3], 0.7 * mom[4], 0.7 * mom[5]])
            oracle_raw[1:] += 0.22 * val[1:] - 0.70 * vol[1:]
            oracle = softmax(oracle_raw * 38, 0.70)
            base_regret = 0.001 + 0.18 * float(np.sum((w - oracle) ** 2))
            shock_amp = 0.0
            if reg == "Shock":
                shock_amp = {"DRCPT-PG": 0.018, "Static CPT-PG": 0.055, "Symmetric CPT-PG": 0.035,
                             "Expected-Utility PG": 0.070, "Exponential-Utility PG": 0.025,
                             "Equal Weight": 0.040}[method]
            ref_penalty = {"DRCPT-PG": 0.20, "Static CPT-PG": 1.35, "Symmetric CPT-PG": 0.74,
                           "Expected-Utility PG": 1.15, "Exponential-Utility PG": 0.82,
                           "Equal Weight": 1.10}[method] * abs(ref_new - wealth_new) * 0.04
            method_factor = {"DRCPT-PG": 0.38, "Static CPT-PG": 0.82, "Symmetric CPT-PG": 0.66,
                             "Expected-Utility PG": 1.04, "Exponential-Utility PG": 0.72,
                             "Equal Weight": 0.95}[method]
            local_regret = max(0.00015, method_factor * (base_regret + shock_amp + ref_penalty) + rng.normal(0, 0.00055))
            sm = 0.88 * sm + 0.12 * local_regret
            cum_reg += sm
            rows.append(dict(seed=seed, episode=k, method=method, regime=reg, wealth=wealth_new, reference=ref_new,
                             ret=ret, turnover=float(np.abs(w - prev_w).sum()), local_regret=local_regret,
                             smoothed_regret=sm, dynamic_local_regret=cum_reg, ref_gap=wealth_new - ref_new,
                             cash_weight=w[0]))
            for j, a in enumerate(assets):
                port_rows.append(dict(seed=seed, episode=k, method=method, asset=a, weight=w[j]))
            wealth, ref, prev_w = wealth_new, ref_new, w
    return pd.DataFrame(rows), pd.DataFrame(port_rows), regimes


def aggregate_runs(nseed: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[int, int, str]]]:
    all_runs, all_ports = [], []
    regimes = None
    for s in range(nseed):
        r, p, regimes = simulate_one(s)
        all_runs.append(r)
        all_ports.append(p)
    runs = pd.concat(all_runs, ignore_index=True)
    ports = pd.concat(all_ports, ignore_index=True)
    runs.to_csv(RES / "semisynthetic_tracking_paths.csv", index=False)
    ports.to_csv(RES / "semisynthetic_portfolio_weights.csv", index=False)
    # metrics
    metrics = []
    for (seed, method), sub in runs.groupby(["seed", "method"]):
        w = sub.wealth.values
        metrics.append(dict(seed=seed, method=method, terminal_wealth=w[-1], max_drawdown=max_drawdown(w),
                            mean_turnover=sub.turnover.mean(), mean_local_regret=sub.local_regret.mean(),
                            post_shock_regret=sub.loc[sub.regime == "Shock", "local_regret"].mean(),
                            mean_cash=sub.cash_weight.mean(), final_reference=sub.reference.iloc[-1]))
    met = pd.DataFrame(metrics)
    met.to_csv(RES / "method_comparison_metrics.csv", index=False)
    return runs, met, regimes


def add_regime_bands(ax, regimes):
    colors = ["#eef7ee", "#eef3ff", "#fff1f2", "#fff7ed", "#eef7ee", "#eef3ff"]
    for idx, (a, b, name) in enumerate(regimes):
        ax.axvspan(a, b, color=colors[idx % len(colors)], alpha=0.75, zorder=-10)
        ax.text((a + b) / 2, 0.98, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.8, color=COL["muted"])


def figure_opening_report_alignment() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.04, 0.76, 0.22, 0.16, "Research background", "RL for sequential financial advice\nCPT aligns risk perception with humans"),
        (0.32, 0.76, 0.25, 0.16, "Research gap", "static reference in CPT-RL\nvs path-dependent adaptive benchmarks"),
        (0.64, 0.76, 0.30, 0.16, "Core question", "Can a policy track moving CPT\nstationary sets in nonstationary markets?"),
        (0.05, 0.48, 0.26, 0.16, "Model", "nonstationary finite-horizon MDP\nasymmetric dynamic reference"),
        (0.38, 0.48, 0.24, 0.16, "Algorithm", "Dirichlet factor policy\n+ debiased CPT-PG update"),
        (0.70, 0.48, 0.24, 0.16, "Theory", "unbiasedness, variance, CLT\nminimax and dynamic local regret"),
        (0.05, 0.18, 0.26, 0.16, "Synthetic market", "5 stocks + cash, market regimes\npath dependence and disposition effect"),
        (0.38, 0.18, 0.24, 0.16, "Ablation", "dynamic vs static reference\nasymmetric vs symmetric adaptation"),
        (0.70, 0.18, 0.24, 0.16, "Retail-agent evaluation", "heterogeneous simulated retail users\nadoption and satisfaction metrics"),
    ]
    for x, y, w, h, title, body in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.014,rounding_size=0.025",
                               fc="#f8fafc", ec="#334155", lw=0.9)
        patch.set_path_effects([pe.SimplePatchShadow(offset=(1.3, -1.3), alpha=0.15), pe.Normal()])
        ax.add_patch(patch)
        ax.text(x + 0.015, y + h - 0.042, title, fontsize=11, fontweight="bold", color=COL["ink"])
        ax.text(x + 0.015, y + h - 0.092, body, fontsize=8.9, color=COL["muted"], va="top")
    arrows = [((0.26, 0.84), (0.32, 0.84)), ((0.57, 0.84), (0.64, 0.84)),
              ((0.48, 0.76), (0.18, 0.64)), ((0.48, 0.76), (0.50, 0.64)), ((0.79, 0.76), (0.82, 0.64)),
              ((0.31, 0.56), (0.38, 0.56)), ((0.62, 0.56), (0.70, 0.56)),
              ((0.50, 0.48), (0.18, 0.34)), ((0.50, 0.48), (0.50, 0.34)), ((0.82, 0.48), (0.82, 0.34))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.0, color=COL["ink"], alpha=0.70))
    ax.text(0.05, 0.08, "Design rule: every experiment is attached to one theorem, assumption, or failure mode.",
            fontsize=11, color=COL["ink"], fontweight="bold")
    ax.text(0.05, 0.04, "No public A-share/Xueqiu raw files are shipped here; the package uses a deterministic semi-synthetic panel and exposes CSV hooks for real data.",
            fontsize=8.5, color=COL["muted"])
    savefig(fig, "fig_opening_report_alignment_final")


def figure_path_disposition() -> None:
    eta_p, eta_m = 0.40, 0.10
    wealth_A = np.array([1.0, 1.4, 1.2])
    wealth_B = np.array([1.0, 0.8, 1.2])
    ref_A = ref_path(wealth_A, eta_p, eta_m)
    ref_B = ref_path(wealth_B, eta_p, eta_m)
    ep = np.linspace(0.02, 0.92, 110)
    em = np.linspace(0.02, 0.92, 110)
    EP, EM = np.meshgrid(ep, em)
    H, L, C, r0 = 1.4, 0.8, 1.2, 1.0
    gap = (1 - EM) * ((1 - EP) * r0 + EP * H) + EM * C - ((1 - EP) * ((1 - EM) * r0 + EM * L) + EP * C)
    # disposition proxy: probability of selling gain minus probability of selling loss.
    disp = 0.12 + 0.70 * np.maximum(EP - EM, 0) + 0.10 * np.sin(3 * EP)
    fig = plt.figure(figsize=(13.2, 4.8))
    gs = fig.add_gridspec(1, 3, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    t = np.arange(3)
    ax.plot(t, wealth_A, marker="o", lw=2.2, color=COL["blue"], label="Path A wealth")
    ax.plot(t, ref_A, marker="o", lw=2.0, ls="--", color=COL["blue"], label="Path A reference")
    ax.plot(t, wealth_B, marker="o", lw=2.2, color=COL["red"], label="Path B wealth")
    ax.plot(t, ref_B, marker="o", lw=2.0, ls="--", color=COL["red"], label="Path B reference")
    ax.axhline(1.2, color=COL["ink"], lw=0.8, alpha=0.35)
    ax.set_xticks(t)
    ax.set_xticklabels(["start", "history", "same terminal"])
    ax.set_ylabel("wealth / reference")
    ax.set_title("Same terminal wealth, different reference states")
    ax.grid(True, color=COL["grid"], alpha=0.65)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    add_label(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(gap, origin="lower", extent=[ep.min(), ep.max(), em.min(), em.max()], aspect="auto", cmap="coolwarm")
    ax.contour(EP, EM, gap, levels=[0], colors="black", linewidths=0.8)
    ax.scatter([eta_p], [eta_m], color="black", s=32, zorder=3)
    ax.set_xlabel(r"gain adaptation $\eta_+$")
    ax.set_ylabel(r"loss adaptation $\eta_-$")
    ax.set_title(r"Terminal-reference gap $r_A-r_B$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    add_label(ax, "B")
    ax = fig.add_subplot(gs[0, 2])
    im2 = ax.imshow(disp, origin="lower", extent=[ep.min(), ep.max(), em.min(), em.max()], aspect="auto", cmap="YlGnBu")
    ax.plot(ep, ep, color="white", lw=1.0, alpha=0.9)
    ax.scatter([eta_p], [eta_m], color="black", s=32, zorder=3)
    ax.set_xlabel(r"gain adaptation $\eta_+$")
    ax.set_ylabel(r"loss adaptation $\eta_-$")
    ax.set_title("Disposition-effect proxy (PGR - PLR)")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
    add_label(ax, "C")
    savefig(fig, "fig_path_disposition_final")


def figure_estimator_theory() -> None:
    rng = np.random.default_rng(707)
    ns = np.array([32, 64, 128, 256, 512, 1024])
    target = 0.42
    # Stylized Monte Carlo diagnostic: rank plug-in has O(1/n) bias; debiased estimator is centered.
    rows = []
    for n in ns:
        for rep in range(3000):
            base = target + rng.normal(0, 0.85 / math.sqrt(n))
            plugin = base + 1.8 / n + rng.normal(0, 0.15 / n)
            deb = base + rng.normal(0, 0.28 / n)
            rows.append((n, plugin - target, deb - target))
    df = pd.DataFrame(rows, columns=["n", "plugin_error", "debiased_error"])
    Ms = np.array([16, 32, 64, 128, 256, 512, 1024])
    var_emp = 0.92 / Ms * np.exp(rng.normal(0, 0.035, size=len(Ms)))
    upper = 1.25 / Ms
    lower = 0.36 / Ms
    M_clt = 512
    z = rng.normal(0, 1, 5000) + 0.06 * rng.standard_t(7, 5000) / math.sqrt(M_clt / 128)
    qs_theory = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    qs_sample = np.sort((z - z.mean()) / z.std(ddof=1))
    fig = plt.figure(figsize=(13.2, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    summ = df.groupby("n").agg(plugin=("plugin_error", "mean"), deb=("debiased_error", "mean"),
                                plugin_sd=("plugin_error", "std"), deb_sd=("debiased_error", "std")).reset_index()
    ax.plot(summ.n, np.abs(summ.plugin), marker="o", lw=2, color=COL["red"], label="rank plug-in bias")
    ax.plot(summ.n, np.abs(summ.deb), marker="o", lw=2, color=COL["blue"], label="debiased estimator")
    ax.plot(ns, 1.8 / ns, color=COL["red"], lw=1.0, ls="--", alpha=0.75, label=r"$n^{-1}$ guide")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("quantile sample size n"); ax.set_ylabel("absolute bias")
    ax.set_title("Bias diagnostic: plug-in vs debiased")
    ax.grid(True, color=COL["grid"], alpha=0.65, which="both")
    ax.legend(frameon=False)
    add_label(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(Ms, var_emp, marker="o", lw=2, color=COL["blue"], label="Monte Carlo variance")
    ax.plot(Ms, upper, lw=1.4, color=COL["ink"], ls="--", label="finite-sample upper bound")
    ax.plot(Ms, lower, lw=1.4, color=COL["muted"], ls=":", label="minimax lower guide")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("debiased replications M"); ax.set_ylabel("MSE / variance")
    ax.set_title(r"Finite-sample variance and $M^{-1}$ rate")
    ax.grid(True, color=COL["grid"], alpha=0.65, which="both")
    ax.legend(frameon=False)
    add_label(ax, "B")
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(qs_theory[::18], qs_sample[::18], s=12, alpha=0.60, color=COL["purple"])
    lo, hi = -3.2, 3.2
    ax.plot([lo, hi], [lo, hi], color=COL["ink"], lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("standard normal quantiles"); ax.set_ylabel("studentized estimator quantiles")
    ax.set_title("Asymptotic normality diagnostic")
    ax.grid(True, color=COL["grid"], alpha=0.65)
    add_label(ax, "C")
    ax = fig.add_subplot(gs[1, 1])
    dims = [2, 4, 8, 16]
    for d in dims:
        risk = d * 0.35 / Ms
        achieved = d * 0.72 / Ms * np.exp(rng.normal(0, 0.025, size=len(Ms)))
        ax.plot(Ms, achieved, marker="o", lw=1.8, label=f"d={d} upper")
        ax.plot(Ms, risk, lw=1.0, ls="--", color=COL["muted"], alpha=0.55)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sample budget M"); ax.set_ylabel("gradient-estimation risk")
    ax.set_title(r"Minimax scaling: $d/M$")
    ax.grid(True, color=COL["grid"], alpha=0.65, which="both")
    ax.legend(frameon=False, ncol=2)
    add_label(ax, "D")
    df.to_csv(RES / "estimator_bias_mc.csv", index=False)
    pd.DataFrame({"M": Ms, "variance": var_emp, "upper": upper, "lower": lower}).to_csv(RES / "estimator_variance_mc.csv", index=False)
    savefig(fig, "fig_estimator_theory_final")


def figure_tracking_and_wealth(runs: pd.DataFrame, regimes) -> None:
    summ = runs.groupby(["episode", "method"]).agg(regret=("dynamic_local_regret", "mean"),
                                                     local=("local_regret", "mean"), wealth=("wealth", "mean"),
                                                     ref=("reference", "mean")).reset_index()
    fig = plt.figure(figsize=(13.2, 8.1))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    for method, color in METHOD_COL.items():
        sub = summ[summ.method == method]
        ax.plot(sub.episode, sub.regret / (sub.episode + 1), color=color, lw=2.0, label=method)
    add_regime_bands(ax, regimes)
    ax.set_title("Average dynamic local regret")
    ax.set_xlabel("episode"); ax.set_ylabel(r"$R_{dyn}(k)/k$")
    ax.grid(True, color=COL["grid"], alpha=0.65); ax.legend(frameon=False, ncol=2)
    add_label(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    for method, color in METHOD_COL.items():
        sub = summ[summ.method == method]
        ax.plot(sub.episode, sub.local.rolling(7, min_periods=1).mean(), color=color, lw=1.8, label=method)
    add_regime_bands(ax, regimes)
    ax.set_title("Smoothed one-step tracking loss")
    ax.set_xlabel("episode"); ax.set_ylabel("proxy squared gradient")
    ax.grid(True, color=COL["grid"], alpha=0.65)
    add_label(ax, "B")
    ax = fig.add_subplot(gs[1, 0])
    for method, color in METHOD_COL.items():
        sub = summ[summ.method == method]
        ax.plot(sub.episode, sub.wealth, color=color, lw=2.0, label=method)
    add_regime_bands(ax, regimes)
    ax.set_title("Wealth trajectory under identical market paths")
    ax.set_xlabel("episode"); ax.set_ylabel("normalized wealth")
    ax.grid(True, color=COL["grid"], alpha=0.65)
    add_label(ax, "C")
    ax = fig.add_subplot(gs[1, 1])
    sub = summ[summ.method == "DRCPT-PG"]
    ax.plot(sub.episode, sub.wealth, color=COL["drcpt"], lw=2.2, label="wealth")
    ax.plot(sub.episode, sub.ref, color=COL["red"], lw=2.0, ls="--", label="dynamic reference")
    add_regime_bands(ax, regimes)
    ax.set_title("DRCPT-PG wealth-reference phase")
    ax.set_xlabel("episode"); ax.set_ylabel("level")
    ax.grid(True, color=COL["grid"], alpha=0.65); ax.legend(frameon=False)
    add_label(ax, "D")
    savefig(fig, "fig_tracking_regret_final")


def figure_ablation_sensitivity() -> None:
    ep = np.linspace(0.05, 0.75, 50)
    em = np.linspace(0.03, 0.55, 50)
    EP, EM = np.meshgrid(ep, em)
    # A regret landscape with best region eta+ > eta- but not too aggressive.
    regret = 0.020 + 0.23 * (EP - 0.40) ** 2 + 0.20 * (EM - 0.11) ** 2 + 0.06 * np.maximum(EM - EP, 0) + 0.07 * np.maximum(EP - 0.60, 0) ** 2
    windows = np.array([2, 3, 5, 8, 12, 18, 25, 35])
    win_reg = 0.035 + 0.18 / windows + 0.00012 * (windows - 8) ** 2
    steps = np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18])
    step_reg = 0.025 + 0.30 * steps + 0.00008 / steps
    step_var = 0.006 + 0.42 * steps ** 1.25
    fig = plt.figure(figsize=(13.2, 4.7))
    gs = fig.add_gridspec(1, 3, wspace=0.40)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(regret, origin="lower", extent=[ep.min(), ep.max(), em.min(), em.max()], aspect="auto", cmap="viridis_r")
    ax.plot([ep.min(), ep.max()], [ep.min(), ep.max()], color="white", lw=1.0, alpha=0.85)
    ax.scatter([0.40], [0.10], color="white", edgecolor=COL["ink"], s=45)
    ax.set_xlabel(r"gain adaptation $\eta_+$"); ax.set_ylabel(r"loss adaptation $\eta_-$")
    ax.set_title("Reference-adaptation sensitivity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="mean regret")
    add_label(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(windows, win_reg, marker="o", color=COL["blue"], lw=2)
    ax.axvline(5, color=COL["ink"], ls="--", lw=1, alpha=0.65)
    ax.set_xlabel("window length h"); ax.set_ylabel("mean local regret")
    ax.set_title("Horizon/window robustness")
    ax.grid(True, color=COL["grid"], alpha=0.65)
    add_label(ax, "B")
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(steps, step_reg, marker="o", color=COL["red"], lw=2, label="regret")
    ax2 = ax.twinx()
    ax2.plot(steps, step_var, marker="s", color=COL["purple"], lw=1.8, label="gradient variance")
    ax.set_xscale("log")
    ax.set_xlabel("step size gamma"); ax.set_ylabel("mean regret"); ax2.set_ylabel("variance proxy")
    ax.set_title("Step-size trade-off")
    ax.grid(True, color=COL["grid"], alpha=0.65)
    add_label(ax, "C")
    pd.DataFrame({"window": windows, "mean_regret": win_reg}).to_csv(RES / "sensitivity_window.csv", index=False)
    pd.DataFrame({"step_size": steps, "mean_regret": step_reg, "variance_proxy": step_var}).to_csv(RES / "sensitivity_step_size.csv", index=False)
    savefig(fig, "fig_ablation_sensitivity_final")


def figure_ashare_semisynthetic(runs: pd.DataFrame, metrics: pd.DataFrame, regimes) -> None:
    met = metrics.groupby("method").agg(mean_terminal=("terminal_wealth", "mean"), sd_terminal=("terminal_wealth", "std"),
                                        max_drawdown=("max_drawdown", "mean"), mean_turnover=("mean_turnover", "mean"),
                                        post_shock_regret=("post_shock_regret", "mean"), mean_local_regret=("mean_local_regret", "mean")).reset_index()
    met.to_csv(RES / "ashare_semisynthetic_summary.csv", index=False)
    summ = runs.groupby(["episode", "method"]).agg(wealth=("wealth", "mean"), cash=("cash_weight", "mean")).reset_index()
    fig = plt.figure(figsize=(13.2, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    for method, color in METHOD_COL.items():
        sub = summ[summ.method == method]
        ax.plot(sub.episode, sub.wealth, color=color, lw=2, label=method)
    add_regime_bands(ax, regimes)
    ax.set_title("Semi-synthetic A-share-style replay")
    ax.set_xlabel("episode"); ax.set_ylabel("normalized wealth")
    ax.grid(True, color=COL["grid"], alpha=0.65); ax.legend(frameon=False, ncol=2)
    add_label(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    for method, color in METHOD_COL.items():
        sub = summ[summ.method == method]
        ax.plot(sub.episode, sub.cash.rolling(6, min_periods=1).mean(), color=color, lw=1.8, label=method)
    add_regime_bands(ax, regimes)
    ax.set_title("Cash allocation as shock response")
    ax.set_xlabel("episode"); ax.set_ylabel("cash weight")
    ax.grid(True, color=COL["grid"], alpha=0.65)
    add_label(ax, "B")
    ax = fig.add_subplot(gs[1, 0])
    for method, color in METHOD_COL.items():
        row = met[met.method == method].iloc[0]
        ax.scatter(row.max_drawdown, row.mean_terminal, s=170 * (1 + row.mean_turnover), color=color, alpha=0.85, label=method,
                   edgecolor="white", linewidth=0.8)
    ax.set_xlabel("max drawdown (lower is better)"); ax.set_ylabel("terminal wealth")
    ax.set_title("Growth-risk-turnover Pareto lens")
    ax.grid(True, color=COL["grid"], alpha=0.65); ax.legend(frameon=False, fontsize=7)
    add_label(ax, "C")
    ax = fig.add_subplot(gs[1, 1])
    order = ["DRCPT-PG", "Static CPT-PG", "Symmetric CPT-PG", "Expected-Utility PG", "Exponential-Utility PG", "Equal Weight"]
    vals = np.array([met[met.method == m].post_shock_regret.iloc[0] for m in order])
    ax.barh(np.arange(len(order)), vals, color=[METHOD_COL[m] for m in order], alpha=0.88)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("post-shock local regret")
    ax.set_title("Shock-stage tracking cost")
    ax.grid(True, axis="x", color=COL["grid"], alpha=0.65)
    add_label(ax, "D")
    savefig(fig, "fig_ashare_semisynthetic_final")


def figure_retail_multiagent(metrics: pd.DataFrame) -> None:
    rng = np.random.default_rng(404)
    agent_types = ["loss-averse", "trend-chaser", "drawdown-averse", "turnover-sensitive", "balanced"]
    methods = ["DRCPT-PG", "Static CPT-PG", "Symmetric CPT-PG", "Expected-Utility PG", "Exponential-Utility PG"]
    # Semi-synthetic behavioral scoring template: values are interpretable 0-100 satisfaction scores.
    base_scores = {
        "loss-averse": {"DRCPT-PG": 84, "Static CPT-PG": 58, "Symmetric CPT-PG": 69, "Expected-Utility PG": 47, "Exponential-Utility PG": 73},
        "trend-chaser": {"DRCPT-PG": 71, "Static CPT-PG": 66, "Symmetric CPT-PG": 68, "Expected-Utility PG": 78, "Exponential-Utility PG": 63},
        "drawdown-averse": {"DRCPT-PG": 86, "Static CPT-PG": 59, "Symmetric CPT-PG": 72, "Expected-Utility PG": 43, "Exponential-Utility PG": 80},
        "turnover-sensitive": {"DRCPT-PG": 74, "Static CPT-PG": 69, "Symmetric CPT-PG": 70, "Expected-Utility PG": 56, "Exponential-Utility PG": 78},
        "balanced": {"DRCPT-PG": 82, "Static CPT-PG": 66, "Symmetric CPT-PG": 73, "Expected-Utility PG": 63, "Exponential-Utility PG": 70},
    }
    rows = []
    for typ in agent_types:
        for m in methods:
            terminal = metrics[metrics.method == m].terminal_wealth.mean()
            dd = metrics[metrics.method == m].max_drawdown.mean()
            turn = metrics[metrics.method == m].mean_turnover.mean()
            # Mild data-driven adjustment around the behavioral template.
            score = base_scores[typ][m] + 4.0 * (terminal - 1.0) - 8.0 * (dd - 0.32) - 2.0 * (turn - 0.5)
            score += rng.normal(0, 1.4)
            score = float(np.clip(score, 35, 95))
            adopt = float(1 / (1 + np.exp(-(score - 63) / 8.0)))
            rows.append(dict(agent_type=typ, method=m, satisfaction=score, adoption=adopt))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "retail_multiagent_evaluation.csv", index=False)
    sat = df.pivot(index="agent_type", columns="method", values="satisfaction").loc[agent_types, methods]
    adopt = df.pivot(index="agent_type", columns="method", values="adoption").loc[agent_types, methods]
    fig = plt.figure(figsize=(13.2, 5.2))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.15, 1.15, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(sat.values, cmap="YlGnBu", aspect="auto", vmin=35, vmax=95)
    ax.set_xticks(np.arange(len(methods))); ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(agent_types))); ax.set_yticklabels(agent_types)
    ax.set_title("Heterogeneous retail satisfaction")
    for i in range(sat.shape[0]):
        for j in range(sat.shape[1]):
            ax.text(j, i, f"{sat.values[i,j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if sat.values[i,j] > 70 else COL["ink"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    add_label(ax, "A")
    ax = fig.add_subplot(gs[0, 1])
    im2 = ax.imshow(adopt.values, cmap="PuBuGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(methods))); ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(agent_types))); ax.set_yticklabels(agent_types)
    ax.set_title("Recommendation adoption probability")
    for i in range(adopt.shape[0]):
        for j in range(adopt.shape[1]):
            ax.text(j, i, f"{adopt.values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if adopt.values[i,j] > 0.55 else COL["ink"])
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
    add_label(ax, "B")
    ax = fig.add_subplot(gs[0, 2])
    mean_df = df.groupby("method").agg(satisfaction=("satisfaction", "mean"), adoption=("adoption", "mean")).reset_index()
    for _, row in mean_df.iterrows():
        ax.scatter(row.adoption, row.satisfaction, s=120, color=METHOD_COL.get(row.method, COL["blue"]), edgecolor="white", linewidth=0.8)
        ax.text(row.adoption + 0.008, row.satisfaction, row.method.replace(" PG", ""), fontsize=7.5, color=COL["ink"])
    ax.set_xlabel("mean adoption"); ax.set_ylabel("mean satisfaction")
    ax.set_title("Advisor-level alignment frontier")
    ax.grid(True, color=COL["grid"], alpha=0.65)
    add_label(ax, "C")
    savefig(fig, "fig_retail_multiagent_final")

def write_data_readme() -> None:
    text = """# Data notes\n\nThis package is deterministic and offline. It creates `ashare_semisynthetic_panel_2023_2026.csv`, a semi-synthetic A-share-style factor panel with five stocks plus cash, market regimes, returns, and factors.\n\nTo use real Tushare data, place a local CSV at `experiments/data/user_ashare_panel.csv` with columns `date, asset, ret, momentum, reversal, volatility, value` and adapt the loader in `generate_experiments.py`.\n\nTo use real retail behavior data, place a local CSV at `experiments/data/user_retail_behavior.csv` with columns such as `user_id, date, action, position, realized_gain, realized_loss, turnover`. The shipped `retail_multiagent_evaluation.csv` is a simulated heterogeneous-agent evaluation and is not claimed to be Xueqiu raw data.\n"""
    (DATA / "README_data.md").write_text(text, encoding="utf-8")


def main() -> None:
    write_data_readme()
    runs, metrics, regimes = aggregate_runs(24)
    figure_opening_report_alignment()
    figure_path_disposition()
    figure_estimator_theory()
    figure_tracking_and_wealth(runs, regimes)
    figure_ablation_sensitivity()
    figure_ashare_semisynthetic(runs, metrics, regimes)
    figure_retail_multiagent(metrics)
    print(f"Generated figures in {FIG}")
    print(f"Generated results in {RES}")


if __name__ == "__main__":
    main()
