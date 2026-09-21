from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

from mvp_cpt_pg.paper_market import PaperPanel, FEATURES
from mvp_cpt_pg.synthetic_env import _semi_synthetic_regime_parameters
from mvp_cpt_pg.paper_experiments import (
    PaperConfig, action_map, next_wealth, policy, policy_features, update_reference,
)
from mvp_cpt_pg.cpt_objective import CPTPreference
from common_semisynthetic import load_panel, standard_config


@dataclass(frozen=True)
class SlowVariationSpec:
    """Theory-aligned continuous environment for E3.

    The population factor-block law is time invariant.  Only the return-law
    coefficients move over a finite interval (stable -> jump -> finite drift ->
    fixed tail).  Endogenous account states are never reset.  Instead, the
    realized episode transition is under-relaxed by alpha_k=k^{-3/4}; hence
    episode-start state increments are O(alpha_k) when the raw transition is
    bounded, and sum_{k<=K} alpha_k = O(K^{1/4})=o(K).
    """

    episodes: int = 300
    horizon: int = 5
    stable_end: int = 40
    abrupt_end: int = 60
    drift_end: int = 120
    state_power: float = 0.75
    return_bound: float = 0.08
    factor_strength: float = 2.5
    execution_seed: int = 2026092301

    def __post_init__(self):
        if not (1 <= self.stable_end < self.abrupt_end < self.drift_end < self.episodes):
            raise ValueError('Require stable < abrupt < drift < episodes')
        if not (0 < self.state_power < 1):
            raise ValueError('state_power must lie in (0,1)')

    def stage(self, episode: int) -> str:
        if episode <= self.stable_end:
            return 'stable'
        if episode <= self.abrupt_end:
            return 'abrupt-change'
        if episode <= self.drift_end:
            return 'gradual-drift'
        return 'slow-tail'

    def state_gain(self, episode: int) -> float:
        if episode < 1:
            raise ValueError('episode must be positive')
        # Tie the diminishing gain to elapsed market time measured in baseline
        # five-day units.  This leaves the h=5 E3 path unchanged and makes E4
        # horizon comparisons operate on the same slow-variation clock.
        equivalent_k = max(1.0, episode * self.horizon / 5.0)
        return float(equivalent_k ** (-self.state_power))

    @staticmethod
    def _as_vector(regime: str) -> tuple[np.ndarray, float, float]:
        coefs, mean, scale = _semi_synthetic_regime_parameters(regime)
        return np.asarray([coefs.get(name, 0.0) for name in FEATURES], dtype=float), float(mean), float(scale)

    def parameters(self, episode: int) -> tuple[np.ndarray, float, float]:
        if episode <= self.stable_end:
            return self._as_vector('uptrend_momentum')
        if episode <= self.abrupt_end:
            return self._as_vector('oscillation_reversal')
        if episode >= self.drift_end:
            return self._as_vector('style_switch_lowvol_quality')

        start, mean0, scale0 = self._as_vector('oscillation_reversal')
        end, mean1, scale1 = self._as_vector('style_switch_lowvol_quality')
        # episode abrupt_end+1 starts exactly from reversal; drift_end reaches terminal law
        denom = self.drift_end - self.abrupt_end - 1
        frac = 0.0 if denom <= 0 else (episode - (self.abrupt_end + 1)) / denom
        frac = float(np.clip(frac, 0.0, 1.0))
        return (1-frac)*start + frac*end, (1-frac)*mean0 + frac*mean1, (1-frac)*scale0 + frac*scale1


class SlowFactorMarket:
    """Fixed empirical factor-block distribution with a slowly varying return law."""

    semi = True

    def __init__(self, panel: PaperPanel, spec: SlowVariationSpec | None = None):
        self.panel = panel
        self.spec = SlowVariationSpec() if spec is None else spec

    def valid_starts(self, horizon: int) -> np.ndarray:
        return np.arange(0, len(self.panel.dates)-horizon+1, dtype=int)

    def sample_starts(self, rng: np.random.Generator, count: int, horizon: int) -> np.ndarray:
        starts = self.valid_starts(horizon)
        return rng.choice(starts, size=int(count), replace=True)

    def draw_returns(self, episode: int, features, rng):
        features = np.asarray(features, dtype=float)
        coefs, mean, scale = self.spec.parameters(episode)
        signal = np.tensordot(features, coefs, axes=([-1], [0]))
        shape = features.shape[:-1]
        common = rng.normal(0.0, 0.006*scale, size=(*shape[:-1], 1))
        noise = rng.normal(0.0, 0.009*scale, size=shape)
        returns = self.spec.return_bound * np.tanh(
            (mean + self.spec.factor_strength*signal + common + noise) / self.spec.return_bound
        )
        returns[..., 0] = 0.0
        return returns

    def execution_episode(self, seed: int, episode: int, horizon: int):
        ss = np.random.SeedSequence([self.spec.execution_seed, int(seed), int(episode), int(horizon)])
        rg_block, rg_ret = [np.random.default_rng(x) for x in ss.spawn(2)]
        start = int(self.sample_starts(rg_block, 1, horizon)[0])
        features = self.panel.features[start:start+horizon].copy()
        tradable = self.panel.tradable[start:start+horizon].copy()
        returns = np.empty((horizon, len(self.panel.codes)), dtype=float)
        for j in range(horizon):
            returns[j] = self.draw_returns(episode, features[j], rg_ret)
            returns[j, ~tradable[j]] = 0.0
        return dict(start=start, features=features, tradable=tradable, returns=returns,
                    dates=self.panel.dates[start:start+horizon])


@dataclass
class SlowEpisodeLaw:
    market: SlowFactorMarket
    episode: int
    wealth: float
    reference: float
    previous: np.ndarray
    config: PaperConfig
    cpt: CPTPreference = CPTPreference()

    def draw(self, theta, rng, count):
        c = self.config
        count = int(count)
        wealth = np.full(count, self.wealth)
        reference = np.full(count, self.reference)
        previous = np.broadcast_to(self.previous, (count, len(self.previous))).copy()
        scores = np.zeros((count, c.dimension))
        starts = self.market.sample_starts(rng, count, c.horizon)
        for offset in range(c.horizon):
            full_features = self.market.panel.features[starts+offset]
            actor_features = policy_features(full_features, previous, c.dimension)
            weights, score = policy(theta, actor_features, rng)
            scores += score
            returns = self.market.draw_returns(self.episode, full_features, rng)
            tradable = self.market.panel.tradable[starts+offset]
            weights = action_map(previous, weights, tradable, c.trade_fraction)
            returns = np.where(tradable, returns, 0.0)
            wealth, _, _ = next_wealth(wealth, previous, weights, returns, c)
            reference = update_reference(wealth, reference, c)
            previous = weights
        return self.cpt.utilities(wealth-reference), scores, wealth


def load_slow_market(spec: SlowVariationSpec | None = None) -> SlowFactorMarket:
    return SlowFactorMarket(load_panel(), SlowVariationSpec() if spec is None else spec)


def slow_config(dimension: int = 10, budget: int = 512,
                gamma: float = 0.08, vartheta: float = 0.05,
                horizon: int = 5, eta_gain: float = 0.20, eta_loss: float = 0.05) -> PaperConfig:
    return replace(standard_config(dimension, budget), gamma=gamma, a0=vartheta,
                   horizon=int(horizon), eta_gain=float(eta_gain), eta_loss=float(eta_loss))


def fixed_context_slow_law(episode: int, dimension: int = 10, budget: int = 512,
                           cpt: CPTPreference | None = None,
                           spec: SlowVariationSpec | None = None) -> SlowEpisodeLaw:
    """E2 fixed-target law in the shared slow-variation market setup.

    E2 begins from the same semi-synthetic setup used by E3 but freezes the
    account context (wealth=reference=1, all cash) so estimator error is
    isolated from online state propagation. The empirical factor-block law and
    episode-specific return law are identical to E3.
    """
    spec = SlowVariationSpec() if spec is None else spec
    market = load_slow_market(spec)
    config = slow_config(dimension, budget, horizon=spec.horizon)
    previous = np.zeros(len(market.panel.codes), dtype=float)
    previous[0] = 1.0
    return SlowEpisodeLaw(market=market, episode=int(episode), wealth=1.0,
                          reference=1.0, previous=previous, config=config,
                          cpt=CPTPreference() if cpt is None else cpt)


def slow_period_names() -> dict[int, str]:
    return {20: 'stable', 50: 'post-change', 90: 'mid-drift', 150: 'post-drift stationary'}


def relax_state(old_wealth: float, old_reference: float, old_previous: np.ndarray,
                raw_wealth: float, raw_reference: float, raw_previous: np.ndarray,
                gain: float):
    """Continuous diminishing-gain episode boundary update.

    Wealth is blended on log scale to preserve positivity; reference and holdings
    are convexly blended.  No state is reset.
    """
    if not 0 < gain <= 1:
        raise ValueError('gain must lie in (0,1]')
    logw = (1-gain)*np.log(old_wealth) + gain*np.log(raw_wealth)
    wealth = float(np.exp(logw))
    reference = float((1-gain)*old_reference + gain*raw_reference)
    previous = (1-gain)*old_previous + gain*raw_previous
    previous = np.maximum(previous, 0.0)
    previous /= previous.sum()
    return wealth, reference, previous
