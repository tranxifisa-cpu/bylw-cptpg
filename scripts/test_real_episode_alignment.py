"""Regression test for date alignment in real-data bootstrap trajectories."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp_cpt_pg.paper_experiments import EpisodeLaw, PaperConfig, action_map
from mvp_cpt_pg.paper_market import PaperMarket, PaperPanel


def main():
    days, assets, dimension = 10, 3, 10
    features = np.zeros((days, assets, dimension))
    features[:, :, 0] = 1.0
    features[:, 0, 1] = 1.0
    features[:, :, 2] = np.arange(days)[:, None]
    returns = np.zeros((days, assets))
    returns[:, 1] = np.arange(days) / 100.0
    tradable = np.ones((days, assets), dtype=bool)
    tradable[:, 2] = np.arange(days) % 2 == 0
    panel = PaperPanel(
        dates=np.array([f"day-{day}" for day in range(days)]),
        codes=["CASH", "A", "B"],
        features=features,
        returns=returns,
        corporate_break=np.zeros((days, assets - 1), dtype=bool),
        tradable=tradable,
    )
    market = PaperMarket(panel, semi=False)
    config = PaperConfig(horizon=2, dimension=dimension, cost=0.0,
                         trade_fraction=1.0)
    law = EpisodeLaw(market, day=8, wealth=1.0, reference=1.0,
                     previous=np.array([1.0, 0.0, 0.0]), config=config)

    seed, count = 17, 6
    expected_rng = np.random.default_rng(seed)
    starts = expected_rng.integers(0, law.day - config.horizon, size=count)
    observed = []
    observed_tradable = []

    def capture_policy(theta, policy_input, rng, sharpness=1.0, concentration_scale=1.0):
        observed.append(policy_input[..., 2].copy())
        weights = np.zeros(policy_input.shape[:-1])
        weights[:, 1] = 1.0
        scores = np.zeros((len(policy_input), dimension))
        return weights, scores

    def capture_action(previous, latent, mask, trade_fraction):
        observed_tradable.append(mask.copy())
        return action_map(previous, latent, mask, trade_fraction)

    with (patch("mvp_cpt_pg.paper_experiments.policy", side_effect=capture_policy),
          patch("mvp_cpt_pg.paper_experiments.action_map", side_effect=capture_action)):
        _, _, terminal = law.draw(
            np.zeros(dimension), np.random.default_rng(seed), count
        )

    np.testing.assert_array_equal(observed[0][:, 0], starts)
    np.testing.assert_array_equal(observed[1][:, 0], starts + 1)
    np.testing.assert_array_equal(observed_tradable[0][:, 2], tradable[starts, 2])
    np.testing.assert_array_equal(observed_tradable[1][:, 2], tradable[starts + 1, 2])
    expected_terminal = (1.0 + starts / 100.0) * (1.0 + (starts + 1) / 100.0)
    np.testing.assert_allclose(terminal, expected_terminal)
    print("real-data bootstrap factors, tradability and returns are date-aligned")


if __name__ == "__main__":
    main()
