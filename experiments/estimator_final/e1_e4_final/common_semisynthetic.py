from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

from mvp_cpt_pg.paper_market import PaperPanel, FEATURES
from mvp_cpt_pg.synthetic_env import _semi_synthetic_regime_parameters
from mvp_cpt_pg.paper_experiments import PaperConfig, EpisodeLaw
from mvp_cpt_pg.cpt_objective import CPTPreference


DEFAULT_PANEL = ROOT / 'inputs' / 'paper_hs300.csv'
RISKY_ASSETS = 30


@dataclass(frozen=True)
class ThreeStageMarketSpec:
    """Continuous three-stage semi-synthetic market used by E2 and E3.

    The historical A-share factor path is traversed chronologically.  Only the
    semi-synthetic return law changes: a stable momentum regime is followed by
    an abrupt switch to the pre-existing reversal regime, then a gradual drift
    from the reversal parameters toward the pre-existing low-vol/value regime.
    All coefficient magnitudes, means, noise scales, bounded-return mapping,
    and actor inputs are inherited from the original semi-synthetic market.
    """

    episodes: int = 120
    horizon: int = 5
    stable_end: int = 40
    abrupt_end: int = 60
    return_bound: float = 0.08
    factor_strength: float = 2.5

    def __post_init__(self):
        if not (1 <= self.stable_end < self.abrupt_end < self.episodes):
            raise ValueError('Require 1 <= stable_end < abrupt_end < episodes')
        if self.horizon < 1 or self.return_bound <= 0 or self.factor_strength <= 0:
            raise ValueError('Invalid three-stage market specification')

    def episode(self, day: int) -> int:
        episode = int(day) // self.horizon + 1
        if not 1 <= episode <= self.episodes:
            raise ValueError('day outside the prescribed 120-episode experiment')
        return episode

    def stage(self, day: int) -> str:
        episode = self.episode(day)
        if episode <= self.stable_end:
            return 'stable'
        if episode <= self.abrupt_end:
            return 'abrupt-change'
        return 'gradual-drift'

    @staticmethod
    def _as_vector(regime: str) -> tuple[np.ndarray, float, float]:
        coefs, mean, scale = _semi_synthetic_regime_parameters(regime)
        return np.asarray([coefs.get(name, 0.0) for name in FEATURES], dtype=float), float(mean), float(scale)

    def parameters(self, day: int) -> tuple[np.ndarray, float, float]:
        episode = self.episode(day)
        if episode <= self.stable_end:
            return self._as_vector('uptrend_momentum')
        if episode <= self.abrupt_end:
            return self._as_vector('oscillation_reversal')

        start, mean0, scale0 = self._as_vector('oscillation_reversal')
        end, mean1, scale1 = self._as_vector('style_switch_lowvol_quality')
        # Episode abrupt_end+1 uses the reversal law exactly; the last episode
        # reaches the style-switch law.  Hence the continuous drift introduces
        # no second jump at the stage boundary.
        denominator = self.episodes - self.abrupt_end - 1
        frac = 0.0 if denominator <= 0 else (episode - (self.abrupt_end + 1)) / denominator
        frac = float(np.clip(frac, 0.0, 1.0))
        return (1 - frac) * start + frac * end, (1 - frac) * mean0 + frac * mean1, (1 - frac) * scale0 + frac * scale1


class ThreeStagePaperMarket:
    """Chronological online market with the original semi-synthetic mechanics."""

    semi = True

    def __init__(self, panel: PaperPanel, spec: ThreeStageMarketSpec | None = None):
        self.panel = panel
        self.spec = ThreeStageMarketSpec() if spec is None else spec
        if len(panel.dates) < self.spec.episodes * self.spec.horizon:
            raise ValueError('Prepared A-share panel is shorter than the E3 horizon')

    def regime(self, day: int) -> str:
        return self.spec.stage(day)

    def draw_returns(self, day: int, features, rng):
        features = np.asarray(features, dtype=float)
        coefs, mean, scale = self.spec.parameters(day)
        signal = np.tensordot(features, coefs, axes=([-1], [0]))
        shape = features.shape[:-1]
        common = rng.normal(0.0, 0.006 * scale, size=(*shape[:-1], 1))
        noise = rng.normal(0.0, 0.009 * scale, size=shape)
        returns = self.spec.return_bound * np.tanh(
            (mean + self.spec.factor_strength * signal + common + noise) / self.spec.return_bound
        )
        returns[..., 0] = 0.0
        return returns

    def execution_returns(self, seed: int) -> np.ndarray:
        """One chronological realized semi-synthetic path, shared across methods."""
        rng = np.random.default_rng(seed)
        total_days = self.spec.episodes * self.spec.horizon
        result = np.stack([
            self.draw_returns(day, self.panel.features[day], rng)
            for day in range(total_days)
        ])
        result[~self.panel.tradable[:total_days]] = 0.0
        return result


def load_panel(input_csv: Path | None = None) -> PaperPanel:
    path = DEFAULT_PANEL if input_csv is None else input_csv
    panel = PaperPanel.load(path, '20230301', '20260529')
    if len(panel.codes) != RISKY_ASSETS + 1:
        raise ValueError(f'E2-E4 require {RISKY_ASSETS} stocks plus cash; got {len(panel.codes) - 1} stocks from {path}')
    return panel


def load_market(input_csv: Path | None = None,
                spec: ThreeStageMarketSpec | None = None) -> ThreeStagePaperMarket:
    return ThreeStagePaperMarket(load_panel(input_csv), spec=spec)


def standard_config(dimension: int = 10, budget: int = 512) -> PaperConfig:
    return PaperConfig(
        horizon=5, gamma=0.05, a0=1.0, theta_radius=2.0,
        eta_gain=0.20, eta_loss=0.05, cost=0.001,
        n=max(1, budget // 8), evaluation_n=1024, evaluation_m=1024,
        window=5, rho=0.9, dimension=dimension,
        trajectory_budget=budget, trade_fraction=0.2,
    )


def episode_start_day(episode: int, spec: ThreeStageMarketSpec | None = None) -> int:
    spec = ThreeStageMarketSpec() if spec is None else spec
    if not 1 <= int(episode) <= spec.episodes:
        raise ValueError('episode outside prescribed environment')
    return (int(episode) - 1) * spec.horizon


def fixed_context_law(episode: int, dimension: int = 10, budget: int = 512,
                      cpt: CPTPreference | None = None,
                      spec: ThreeStageMarketSpec | None = None) -> EpisodeLaw:
    """E2 fixed-target law at a pre-specified time in the same E3 environment.

    E2 standardizes the portfolio context to wealth=reference=1 and all cash so
    estimator error is isolated from endogenous account histories.  The market
    date, historical factor vector, return law, policy class, and CPT target are
    exactly the ones used by E3 at that time.
    """
    spec = ThreeStageMarketSpec() if spec is None else spec
    market = load_market(spec=spec)
    config = standard_config(dimension, budget)
    previous = np.zeros(len(market.panel.codes), dtype=float)
    previous[0] = 1.0
    return EpisodeLaw(
        market=market,
        day=episode_start_day(episode, spec),
        wealth=1.0,
        reference=1.0,
        previous=previous,
        config=config,
        cpt=CPTPreference() if cpt is None else cpt,
    )


def period_names() -> dict[int, str]:
    return {20: 'stable', 50: 'post-change', 90: 'mid-drift', 150: 'post-drift stationary'}
