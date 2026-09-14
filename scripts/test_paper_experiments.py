"""Small fixtures only; no Tushare access and no formal experiment runs."""

from dataclasses import replace
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import binom

from mvp_cpt_pg.paper_market import PaperMarket, PaperPanel, dated_cache, prepare_panel
from mvp_cpt_pg.paper_experiments import (EpisodeLaw, PaperConfig, method_config,
    executable_target, multilevel, next_wealth, objective, policy, projected_update, run_online,
    budgeted_gradient, improved_multilevel, estimator_cost)
from mvp_cpt_pg.controlled_cpt import SmoothCPT, quantile_weight
from mvp_cpt_pg.paper_figures import plot_experiment
from run_paper_experiments import save_online


def fixture():
    features = np.zeros((12, 4, 10))
    features[:, :, 0] = 1
    features[:, 0, 1] = 1
    features[:, 1:, 2] = [-.4, 0, .4]
    features[:, 1:, 4] = [.3, -.2, -.1]
    return PaperMarket(PaperPanel(np.array([f"day{i:02}" for i in range(12)]),
        ["CASH", "A", "B", "C"], features, np.zeros((12, 4)), np.zeros((12, 3), bool)), True)


class PaperExperimentTests(unittest.TestCase):
    def test_improved_estimators_match_exact_finite_level_expectation(self):
        class BinaryLaw:
            config = PaperConfig(base=2, cap=2, dimension=2, outer_batch=3)
            previous = np.array([1., 0])
            cpt = SmoothCPT()
            def draw(self, theta, rng, count):
                coin = rng.binomial(1, .4, count)
                utilities = np.c_[coin * .2, (1 - coin) * .3]
                score = (coin - .4)[:, None] * np.array([1., 2.])
                return utilities, score, 1 + .1 * coin
        law = BinaryLaw()
        n = law.config.base * 2**law.config.cap
        exact = np.zeros(2)
        for k in range(n + 1):
            inner = np.array([[.2, 0]] * k + [[0, .3]] * (n - k))
            outer = np.array([[0, .3], [.2, 0]])
            multiplier = quantile_weight(inner, outer, law.cpt)
            exact += binom.pmf(k, n, .4) * np.sum(multiplier * np.array([.6, .4])
                                                        * np.array([-.4, .6])) * np.array([1., 2.])
        for variant in ["multilevel_outer", "split_base"]:
            values, costs = improved_multilevel(law, np.zeros(2), np.random.default_rng(82), 16000, variant)
            error = abs(values.mean(axis=0) - exact)
            self.assertTrue(np.all(error < 6 * values.std(axis=0, ddof=1) / np.sqrt(len(values))))
            self.assertLess(abs(costs.mean() - estimator_cost(law.config, variant)), .15)

    def test_budget_and_common_probe(self):
        config = PaperConfig(horizon=2, base=2, cap=1, n=4, outer_batch=2,
                             trajectory_budget=32, evaluation_n=8, evaluation_m=6)
        law = EpisodeLaw(fixture(), 0, 1., 1., np.array([1., 0, 0, 0]), config)
        _, cost = budgeted_gradient(law, np.zeros(10), np.random.default_rng(1), "plugin", 32)
        self.assertEqual(cost, 32)
        with self.assertRaisesRegex(ValueError, "exceed"):
            budgeted_gradient(law, np.zeros(10), np.random.default_rng(1), "plugin", 4)
        trace, _ = run_online(fixture(), config, [9], ["dynamic", "frozen"], 4,
                              estimator="split_base", common_probe=True)
        first = trace[trace.episode == 1]
        np.testing.assert_allclose(first.common_objective, first.common_objective.iloc[0])
        self.assertTrue((trace[trace.method == "frozen"].training_trajectories == 0).all())
        self.assertTrue(np.isfinite(trace.common_gradient_cross).all())

    def test_dated_cache_field_variants_and_conflicts(self):
        with TemporaryDirectory(prefix="paper_cache_variants_") as directory:
            folder = Path(directory) / "daily_basic"
            folder.mkdir()
            full = pd.DataFrame(dict(ts_code=["A", "B"], trade_date="20260108",
                                     pb=[1., 2.], circ_mv=[100., 200.]))
            def write(name, frame, fields=True):
                payload = {"trade_date": "20260108"}
                if fields:
                    payload["fields"] = ",".join(frame.columns)
                (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
                frame.to_pickle(folder / f"{name}.pkl")
            required = ("ts_code", "trade_date", "pb", "circ_mv")
            write("thin", full.drop(columns="circ_mv"))
            with self.assertRaisesRegex(ValueError, "required columns"):
                dated_cache(directory, "daily_basic", required)
            write("full", full)
            self.assertEqual(dated_cache(directory, "daily_basic", required)["20260108"].name, "full.pkl")
            write("extra", full.iloc[::-1].assign(pe=12.))
            self.assertEqual(dated_cache(directory, "daily_basic", required)["20260108"].name, "extra.pkl")
            write("unknown_schema", full.drop(columns="circ_mv"), fields=False)
            self.assertEqual(dated_cache(directory, "daily_basic", required)["20260108"].name, "extra.pkl")
            write("conflict", full.assign(pb=[9., 2.]))
            with self.assertRaisesRegex(ValueError, "Conflicting daily_basic cache data for 20260108"):
                dated_cache(directory, "daily_basic", required)

    def test_dated_cache_rejects_invalid_duplicate_keys(self):
        with TemporaryDirectory(prefix="paper_cache_keys_") as directory:
            folder = Path(directory) / "daily"
            folder.mkdir()
            for name, codes in [("good", ["A", "B"]), ("bad", ["A", "A"])]:
                (folder / f"{name}.json").write_text('{"trade_date":"20260108"}', encoding="utf-8")
                pd.DataFrame(dict(ts_code=codes, trade_date="20260108", close=[1., 2.])).to_pickle(folder / f"{name}.pkl")
            with self.assertRaisesRegex(ValueError, "Invalid daily cache keys"):
                dated_cache(directory, "daily", ("ts_code", "trade_date", "close"))

    def test_score_matches_density_derivative(self):
        features = fixture().panel.features[0, :, :4]
        theta = np.array([.1, .2, -.1, .3])
        weights, score = policy(theta, features, np.random.default_rng(3))
        def log_density(t):
            a = np.exp(features @ t)
            return gammaln(a.sum()) - gammaln(a).sum() + ((a - 1) * np.log(weights)).sum()
        finite = []
        for i in range(len(theta)):
            eps = np.eye(len(theta))[i] * 1e-6
            finite.append((log_density(theta + eps) - log_density(theta - eps)) / 2e-6)
        np.testing.assert_allclose(score, finite, rtol=1e-6, atol=1e-8)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertTrue(np.all(weights >= 0))

    def test_mean_and_sampling_expectation(self):
        q = fixture().panel.features[0, :, :4]
        t = np.array([.1, .2, -.1, .3])
        mean, _ = policy(t, q)
        sampled, scores = policy(t, np.broadcast_to(q, (30000, *q.shape)), np.random.default_rng(8))
        np.testing.assert_allclose(sampled.mean(axis=0), mean, atol=.006)
        np.testing.assert_allclose(scores.mean(axis=0), 0, atol=.022)

    def test_projected_update_norm_and_ball(self):
        c = PaperConfig(theta_radius=.5, gamma=.05)
        theta = np.zeros(10)
        result = projected_update(theta, np.full(10, 10.), c)
        self.assertAlmostEqual(np.linalg.norm(result), c.gamma)
        result = projected_update(np.full(10, .5 / np.sqrt(10)), np.ones(10), c)
        self.assertAlmostEqual(np.linalg.norm(result), .5)

    def test_cost_and_cash(self):
        c = PaperConfig(cost=.001)
        wealth, fee, turnover = next_wealth(1., np.array([1., 0]), np.array([0., 1.]),
                                           np.array([0., .1]), c)
        self.assertAlmostEqual(wealth, 1.098)
        self.assertAlmostEqual(fee, .002)
        self.assertAlmostEqual(turnover, 1.)

    def test_untradable_position_is_frozen(self):
        previous = np.array([.2, .5, .3])
        target = np.array([.1, .1, .8])
        result = executable_target(previous, target, np.array([True, False, True]))
        self.assertAlmostEqual(result[1], previous[1])
        self.assertAlmostEqual(result.sum(), 1.)
        self.assertTrue(np.all(result >= 0))

    def test_objective_uses_w(self):
        cpt = SmoothCPT()
        utilities = np.array([[0, .4], [.2, 0]])
        self.assertAlmostEqual(objective(utilities, cpt), (.2 - .4) * cpt.weight(.5))

    def test_multilevel_cost_and_saved_context(self):
        c = PaperConfig(horizon=2, cap=2, m=4)
        law = EpisodeLaw(fixture(), 0, 1., 1., np.array([1., 0, 0, 0]), c)
        previous = law.previous.copy()
        values, costs = multilevel(law, np.zeros(10), np.random.default_rng(4), 8)
        self.assertEqual(values.shape, (8, 10))
        self.assertTrue(set(costs).issubset({3, 5, 9}))
        np.testing.assert_array_equal(previous, law.previous)
        self.assertEqual(law.wealth, 1.)

    def test_cap_zero_coupling_with_constant_law(self):
        class ConstantLaw:
            config = PaperConfig(cap=0, base=2)
            previous = np.array([1., 0])
            cpt = SmoothCPT()
            def draw(self, theta, rng, count):
                return np.tile([.2, 0], (count, 1)), np.ones((count, 10)), np.ones(count)
        values, costs = multilevel(ConstantLaw(), np.zeros(10), np.random.default_rng(3), 5)
        np.testing.assert_allclose(values, 0, atol=1e-15)
        np.testing.assert_array_equal(costs, 3)

    def test_real_simulator_never_reads_future_returns(self):
        market = fixture()
        market.semi = False
        c = PaperConfig(horizon=2)
        law = EpisodeLaw(market, 5, 1., 1., np.array([1., 0, 0, 0]), c)
        before = law.draw(np.zeros(10), np.random.default_rng(1), 8)
        market.panel.returns[4:, 1:] = .5
        market.panel.features[6:] = .8
        after = law.draw(np.zeros(10), np.random.default_rng(1), 8)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_online_metrics_and_reference_ablation(self):
        c = PaperConfig(horizon=2, m=2, n=4, cap=1, evaluation_n=8, evaluation_m=6, window=2)
        trace, daily = run_online(fixture(), c, [8], ["dynamic", "static", "equal_weight"], 4)
        self.assertEqual(len(trace), 6)
        self.assertEqual(len(daily), 12)
        self.assertTrue((trace[trace.method == "static"].reference == 1).all())
        self.assertTrue(trace[trace.method == "equal_weight"].q_squared.isna().all())
        q = trace[trace.method == "dynamic"][[f"q_{i}" for i in range(10)]].to_numpy()
        expected = q[0] @ q[0] + np.sum(((q[1] + c.rho * q[0]) / (1 + c.rho)) ** 2)
        self.assertAlmostEqual(trace[trace.method == "dynamic"].dynamic_local_regret.iloc[-1], expected)
        self.assertEqual(method_config(c, "symmetric").eta_gain, (c.eta_gain + c.eta_loss) / 2)

    def test_execution_uses_old_theta_until_next_episode(self):
        c = PaperConfig(horizon=2, m=2, cap=1, evaluation_n=8, evaluation_m=6)
        _, daily = run_online(fixture(), c, [9], ["dynamic"], 4)
        np.testing.assert_allclose(daily[daily.episode == 1].cash, .25)

    def test_offline_panel_selection_and_lag(self):
        with TemporaryDirectory(prefix="paper_cache_test_") as directory:
            root = Path(directory)
            dates = pd.bdate_range("2023-01-02", periods=25).strftime("%Y%m%d").tolist()
            def write(namespace, name, payload, frame):
                folder = root / namespace
                folder.mkdir(exist_ok=True)
                (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
                frame.to_pickle(folder / f"{name}.pkl")
            write("index_weight", "initial", {"index_code": "000300.SH"},
                  pd.DataFrame(dict(trade_date=["20221230"] * 3, con_code=["A", "B", "C"])))
            write("trade_cal", "calendar", {}, pd.DataFrame(dict(cal_date=dates, is_open=1)))
            for index, date in enumerate(dates):
                close = np.array([10., 20., 30.]) * (1.001 ** index)
                frame = pd.DataFrame(dict(ts_code=["A", "B", "C"], trade_date=date,
                    open=close, close=close, pre_close=close / 1.001, vol=10., pb=[1., 2., 3.],
                    turnover_rate=1., volume_ratio=1., circ_mv=[100., 200., 300.]))
                write("daily", date, {"trade_date": date}, frame.iloc[:, :6])
                write("daily_basic", date, {"trade_date": date}, frame[["ts_code", "trade_date", "pb", "turnover_rate", "volume_ratio", "circ_mv"]])
            output = root / "panel.csv"
            metadata = prepare_panel(root, output, dates[21], dates[-1], 2, 7)
            panel = PaperPanel.load(output, dates[21], dates[-1])
            self.assertEqual(panel.features.shape, (4, 3, 10))
            self.assertLess(metadata["snapshot"], dates[21])
            quote = pd.read_pickle(root / "daily" / f"{dates[-1]}.pkl")
            quote = quote[quote.ts_code != metadata["assets"][0]]
            quote.to_pickle(root / "daily" / f"{dates[-1]}.pkl")
            with self.assertRaisesRegex(ValueError, "Fixed pool has missing data"):
                prepare_panel(root, root / "second.csv", dates[21], dates[-1], 2, 7)
            carried = prepare_panel(root, root / "carried.csv", dates[21], dates[-1], 2, 7,
                                    allow_suspension_carry=True)
            carried_panel = PaperPanel.load(root / "carried.csv", dates[21], dates[-1])
            self.assertTrue(carried["suspension_dates"])
            missing_code = metadata["assets"][0]
            missing_index = carried_panel.codes.index(missing_code)
            self.assertFalse(carried_panel.tradable[-1, missing_index])
            self.assertEqual(carried_panel.returns[-1, missing_index], 0.)

    def test_saved_tracking_plot_exports(self):
        c = PaperConfig(horizon=2, m=2, cap=0, evaluation_n=8, evaluation_m=6)
        trace, daily = run_online(fixture(), c, [8], ["dynamic", "equal_weight"], 4)
        with TemporaryDirectory(prefix="paper_plot_test_") as directory:
            output = Path(directory)
            save_online(output, trace, daily)
            plot_experiment(output, 3)
            plot_experiment(output, 5)
            for name in ["experiment3_tracking_merged", "experiment5_real_quote_replay", "tracking_dynamic"]:
                for suffix in ["pdf", "svg", "png"]:
                    self.assertGreater((output / f"{name}.{suffix}").stat().st_size, 1000)

    def test_other_panels_render_from_fixture_tables(self):
        with TemporaryDirectory(prefix="paper_panels_test_") as directory:
            output = Path(directory)
            pd.DataFrame([dict(path=name, time=t, wealth=wealth, reference=reference)
                for name, values in [("gain_first", [1., 1.4, 1.2]), ("loss_first", [1., .8, 1.2])]
                for t, (wealth, reference) in enumerate(zip(values, [1., 1.1, 1.15]))]).to_csv(output / "paths.csv", index=False)
            pd.DataFrame([dict(eta_gain=g, eta_loss=l, reference_gap=g - l)
                for g in [0., .2] for l in [0., .2]]).to_csv(output / "path_grid.csv", index=False)
            pd.DataFrame([dict(n=n, estimator=method, bias_coordinate=.1 / n, bias_se=.01)
                for n in [2, 4] for method in ["plugin", "multilevel"]]).to_csv(output / "bias_summary.csv", index=False)
            pd.DataFrame([dict(dimension=d, M=m, variance_trace=.1 / m, cost=3 * m, mse=.2 / m)
                for d in [2, 4] for m in [2, 4]]).to_csv(output / "variance_summary.csv", index=False)
            pd.DataFrame([dict(dimension=4, M=4, g0=g) for g in [.1, -.1, .03, .05]]).to_csv(output / "variance_raw.csv", index=False)
            sensitivity = [dict(sweep="eta", eta_gain=g, eta_loss=l, average_dlr=.1)
                           for g in [0., .2] for l in [0., .2]]
            sensitivity += [dict(sweep="horizon", horizon=h, average_dlr=.1) for h in [2, 5]]
            sensitivity += [dict(sweep="gamma", gamma=g, average_dlr=.1) for g in [.01, .05]]
            pd.DataFrame(sensitivity).to_csv(output / "sensitivity.csv", index=False)
            for experiment in [1, 2, 4]:
                plot_experiment(output, experiment)
            self.assertEqual(len(list(output.glob("*.pdf"))), 3)


if __name__ == "__main__":
    unittest.main()
