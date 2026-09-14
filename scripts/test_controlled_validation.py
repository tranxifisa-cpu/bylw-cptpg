"""Deterministic and stochastic checks for the controlled validation core."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from mvp_cpt_pg.controlled_cpt import (
    BetaMarket, SmoothCPT, capped_estimates, expected_cost, paired_quantile_weight,
    projected_update, quantile_weight, residual, residual_metrics, stationary_points,
)
from summarize_controlled_validation import check_error_summary, required_artifacts, verify


class ControlledValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "output_paper/DRCPT_PG_Third_Optimized/verification_experiments.py"
        spec = importlib.util.spec_from_file_location("released_verification", path)
        cls.old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.old)

    def test_endpoint_normalization_and_utility_zero(self):
        cpt = SmoothCPT()
        self.assertEqual(float(cpt.weight(0)), 0)
        self.assertAlmostEqual(float(cpt.weight(1)), 1, places=14)
        np.testing.assert_array_equal(cpt.utilities(0), [0, 0])

    def test_quantile_weight_matches_released_integrator(self):
        rng = np.random.default_rng(2)
        for n in (1, 2, 7, 128):
            inner = rng.choice([0.0, 0.01, 0.04, 0.1], size=(n, 2))
            outer = np.array([[0, 0], [0.01, 0.1], [0.4, 0.3]])
            expected = [self.old.T(inner, target, 1) for target in outer]
            np.testing.assert_allclose(quantile_weight(inner, outer), expected, atol=1e-14)

    def test_batched_quantiles_and_coupling(self):
        rng = np.random.default_rng(3)
        inner = rng.random((19, 16, 2))
        outer = rng.random((19, 2))
        actual = paired_quantile_weight(inner, outer, SmoothCPT())
        expected = [quantile_weight(x, y[None, :])[0] for x, y in zip(inner, outer)]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        fine = actual
        half = (paired_quantile_weight(inner[:, :8], outer, SmoothCPT())
                + paired_quantile_weight(inner[:, 8:], outer, SmoothCPT())) / 2
        old = [self.old.coupled(x, y, 1)[1] for x, y in zip(inner, outer)]
        np.testing.assert_allclose(fine - half, old, atol=1e-14)

    def test_oracle_matches_objective_finite_difference(self):
        for p in (0.0, 0.3, 0.6, 1.0):
            for r in (0.98, 1.0, 1.02):
                market = BetaMarket(probability=p, reference=r)
                for theta in (-1.0, 0.2, 1.0):
                    delta = 1e-5
                    actual = market.oracle(theta, 256)[1]
                    expected = (market.oracle(theta + delta, 256)[0]
                                - market.oracle(theta - delta, 256)[0]) / (2 * delta)
                    self.assertAlmostEqual(actual, expected, delta=2e-7)

    def test_oracle_matches_released_model(self):
        for theta in (-1, 0, 1):
            market = BetaMarket(reference=1.02)
            np.testing.assert_allclose(market.oracle(theta, 256),
                                       self.old.exact_episode(theta, "full", 1, 1.02, .6, 256), atol=1e-13)

    def test_sample_score_moments_and_cost(self):
        market = BetaMarket()
        _, score = market.draw(np.random.default_rng(14), 0.2, 200000)
        self.assertLess(abs(score.mean()), 0.01)
        self.assertLess(abs(np.mean(score**2) - 1), 0.03)
        estimates, costs = capped_estimates(market, 0, np.random.default_rng(1), 100000)
        self.assertTrue(np.isfinite(estimates).all())
        self.assertLess(abs(costs.mean() - expected_cost()), 5 * costs.std() / np.sqrt(len(costs)))
        self.assertTrue(np.all(np.isin(costs, 1 + 2 * 2**np.arange(8))))

    def test_update_uses_threshold_and_projection(self):
        self.assertAlmostEqual(projected_update(0, 0.01, 0.2), 0.002)
        self.assertAlmostEqual(projected_update(0, 10, 0.2), 0.2)
        self.assertEqual(projected_update(1.9, 10, 0.2), 2)
        self.assertEqual(residual(2, 10, 0.2), 0)
        self.assertLess(residual(2, -0.01, 0.2), 0)

    def test_dlr_is_cumulative_squared_smoothed_residual(self):
        q = np.array([1, -1, 1, -1.0])
        smooth, cumulative, average, avg_q = residual_metrics(q, window=2, rho=1)
        np.testing.assert_array_equal(smooth, [1, 0, 0, 0])
        np.testing.assert_array_equal(cumulative, [1, 1, 1, 1])
        np.testing.assert_allclose(average, [1, .5, 1/3, .25])
        np.testing.assert_array_equal(avg_q, np.ones(4))
        np.testing.assert_array_equal(residual_metrics(q, window=1)[1], [1, 2, 3, 4])

    def test_constrained_stationary_endpoints(self):
        np.testing.assert_array_equal(stationary_points(BetaMarket(probability=0.3)), [-2.0])
        np.testing.assert_array_equal(stationary_points(BetaMarket(probability=0.7)), [2.0])

    def test_invalid_utility_and_update_inputs(self):
        with self.assertRaises(ValueError):
            quantile_weight(np.empty((0, 2)), np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            quantile_weight(np.array([[-1, 0]]), np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            projected_update(0, 1, 0)

    def test_missing_experiment_artifacts_are_rejected(self):
        for suite in ("estimator", "tracking", "path", "all"):
            self.assertGreater(len(required_artifacts(suite)), 0)
            with TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / "config.json").write_text(json.dumps({"suite": suite}), encoding="utf-8")
                (directory / "completion.json").write_text(
                    json.dumps({"status": "complete", "suite": suite}), encoding="utf-8")
                with self.assertRaises(FileNotFoundError):
                    verify(directory)
                self.assertFalse((directory / "numerical_checks.json").exists())

    def test_estimator_error_bars_are_verified(self):
        values = np.array([0.0, 1.0, 2.0])
        row = dict(mean=1.0, bias=1.0, mean_se=np.sqrt(1 / 3), variance=1.0,
                   mse=5 / 3, mse_se=np.sqrt(13 / 9))
        check_error_summary(values, 0.0, row)
        for field in row:
            with self.subTest(field=field):
                corrupted = {**row, field: row[field] + 0.1}
                with self.assertRaises(AssertionError):
                    check_error_summary(values, 0.0, corrupted)
        with self.assertRaises(AssertionError):
            check_error_summary([0.0, float("nan"), 2.0], 0.0, row)


if __name__ == "__main__":
    unittest.main()
