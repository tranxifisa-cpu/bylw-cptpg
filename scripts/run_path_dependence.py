from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvp_cpt_pg.config import ExperimentConfig
from mvp_cpt_pg.synthetic_env import SyntheticCPTPGTracker, SyntheticMarket, SyntheticMarketConfig
from mvp_cpt_pg.strategies import update_reference_point
from mvp_cpt_pg.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run path-dependence diagnostics for wealth-space dynamic CPT-PG.")
    parser.add_argument("--policy-normalizer", choices=["softmax", "sparsemax", "dirichlet"], default="dirichlet")
    parser.add_argument(
        "--methods",
        choices=["dynamic_cpt_pg", "static_cpt_pg"],
        nargs="+",
        default=["dynamic_cpt_pg", "static_cpt_pg"],
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--prewarm-steps", type=int, default=30)
    parser.add_argument("--risky-assets", type=int, default=30)
    parser.add_argument("--evaluation-horizon", type=int, default=15)
    parser.add_argument("--cpt-samples", type=int, default=256)
    parser.add_argument("--gradient-samples", type=int, default=32)
    parser.add_argument(
        "--shared-cpt-gradient-samples",
        action="store_true",
        help="Use the same trajectories for CPT quantiles/objective and score-function gradient estimation",
    )
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--eta-gain", type=float, default=0.4)
    parser.add_argument("--eta-loss", type=float, default=0.1)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--return-bound", type=float, default=0.08)
    parser.add_argument("--factor-strength", type=float, default=2.5)
    parser.add_argument(
        "--dirichlet-execution-mode",
        choices=["sample", "mean"],
        default="mean",
        help="Dirichlet mode for actual episode execution; gradient/objective estimation always samples",
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/synthetic"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        evaluation_horizon=args.evaluation_horizon,
        cpt_sample_base=args.cpt_samples,
        gradient_sample_base=args.gradient_samples,
        shared_cpt_gradient_samples=args.shared_cpt_gradient_samples,
        fixed_sample_counts=True,
        gamma0=args.gamma0,
        gamma_exponent=0.0,
        policy_normalizer=args.policy_normalizer,
        dirichlet_execution_mode=args.dirichlet_execution_mode,
        eta_gain=args.eta_gain,
        eta_loss=args.eta_loss,
        trade_cost_bps=args.transaction_cost_bps,
    )
    market = SyntheticMarket.build(
        SyntheticMarketConfig(
            steps=args.steps,
            prewarm_steps=args.prewarm_steps,
            risky_assets=args.risky_assets,
            seed=args.seed,
            return_bound=args.return_bound,
            transaction_cost_bps=args.transaction_cost_bps,
            factor_strength=args.factor_strength,
        )
    )
    path_states = build_path_states(args.eta_gain, args.eta_loss)
    rows: list[pd.DataFrame] = []
    for path_name, state in path_states.items():
        for method in args.methods:
            tracker = SyntheticCPTPGTracker(
                market=market,
                config=config,
                policy=args.policy_normalizer,
                seed=args.seed,
                method=method,
                dirichlet_execution_mode=args.dirichlet_execution_mode,
            )
            initial_reference = state["reference_point"] if method == "dynamic_cpt_pg" else 1.0
            tracker.set_state(
                normalized_wealth=state["normalized_wealth"],
                reference_point=initial_reference,
            )
            trace = tracker.run()
            trace["method"] = method
            trace["path_type"] = path_name
            trace["initial_path_normalized_wealth"] = state["normalized_wealth"]
            trace["initial_path_reference_point"] = initial_reference
            trace["historical_dynamic_reference_point"] = state["reference_point"]
            rows.append(trace)
    result = pd.concat(rows, ignore_index=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(args.output_root / f"path_dependence_{stamp}_{args.policy_normalizer}_seed{args.seed}")
    trace_path = output_dir / "path_dependence_trace.csv"
    result.to_csv(trace_path, index=False, encoding="utf-8-sig")
    summary = summarize_path_dependence(result)
    summary_path = output_dir / "path_dependence_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_path = save_path_dependence_plot(result, output_dir / "path_dependence_diagnostics.png")
    gap_plot_path = save_dynamic_static_gap_plot(result, output_dir / "dynamic_minus_static.png")
    print(f"trace: {trace_path}")
    print(f"summary: {summary_path}")
    print(f"plot: {plot_path}")
    print(f"gap_plot: {gap_plot_path}")


def build_path_states(eta_gain: float, eta_loss: float) -> dict[str, dict[str, float]]:
    paths = {
        "peak_then_crash": [0.02] * 30 + [0.01] * 20 + [0.0] * 20 + [-0.03] * 30,
        "crash_then_recovery": [-0.03] * 30 + [-0.01] * 20 + [0.0] * 20 + [0.035] * 30,
    }
    states: dict[str, dict[str, float]] = {}
    for name, returns in paths.items():
        wealth = 1.0
        reference = 1.0
        for ret in returns:
            wealth *= 1.0 + ret
            reference = update_reference_point(reference, wealth, eta_gain, eta_loss)
        states[name] = {"normalized_wealth": wealth, "reference_point": reference}
    target = min(state["normalized_wealth"] for state in states.values())
    for state in states.values():
        state["normalized_wealth"] = target
    return states


def summarize_path_dependence(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, path_type), frame in trace.groupby(["method", "path_type"]):
        early = frame[frame["step"].between(1, 30)]
        late = frame[frame["step"].between(max(1, frame["step"].max() - 29), frame["step"].max())]
        rows.append(
            {
                "method": method,
                "path_type": path_type,
                "initial_normalized_wealth": float(frame["initial_path_normalized_wealth"].iloc[0]),
                "initial_reference_point": float(frame["initial_path_reference_point"].iloc[0]),
                "historical_dynamic_reference_point": float(frame["historical_dynamic_reference_point"].iloc[0]),
                "early_risky_weight": float((1.0 - early["cash_weight"]).mean()),
                "late_risky_weight": float((1.0 - late["cash_weight"]).mean()),
                "early_relative_wealth": float(early["relative_wealth"].mean()),
                "late_relative_wealth": float(late["relative_wealth"].mean()),
                "early_disposition_spread": float(early["disposition_spread"].mean()),
                "late_disposition_spread": float(late["disposition_spread"].mean()),
                "final_normalized_wealth": float(frame["normalized_wealth"].iloc[-1]),
                "final_reference_point": float(frame["reference_point"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def save_path_dependence_plot(trace: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("normalized_wealth", "Normalized Wealth"),
        ("reference_point", "Reference Point"),
        ("relative_wealth", "Relative Wealth"),
        ("cash_weight", "Cash Weight"),
        ("disposition_spread", "Disposition Spread"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 11), sharex=True)
    colors = {"dynamic_cpt_pg": "#D55E00", "static_cpt_pg": "#0072B2"}
    linestyles = {"peak_then_crash": "-", "crash_then_recovery": "--"}
    for ax, (column, title) in zip(axes, panels):
        for (method, path_type), frame in trace.groupby(["method", "path_type"]):
            label = f"{method} | {path_type}"
            ax.plot(
                frame["step"],
                frame[column],
                label=label,
                linewidth=1.25,
                color=colors.get(method),
                linestyle=linestyles.get(path_type, "-"),
            )
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Synthetic evaluation step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_dynamic_static_gap_plot(trace: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("normalized_wealth", "Dynamic minus Static: Normalized Wealth"),
        ("objective_estimate", "Dynamic minus Static: Objective Estimate"),
        ("cash_weight", "Dynamic minus Static: Cash Weight"),
        ("disposition_spread", "Dynamic minus Static: Disposition Spread"),
    ]
    dynamic = trace[trace["method"] == "dynamic_cpt_pg"]
    static = trace[trace["method"] == "static_cpt_pg"]
    if dynamic.empty or static.empty:
        return output_path
    merged = dynamic.merge(
        static,
        on=["path_type", "step", "seed"],
        suffixes=("_dynamic", "_static"),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 9), sharex=True)
    colors = {"peak_then_crash": "#D55E00", "crash_then_recovery": "#0072B2"}
    for ax, (column, title) in zip(axes, panels):
        for path_type, frame in merged.groupby("path_type"):
            gap = frame[f"{column}_dynamic"] - frame[f"{column}_static"]
            ax.plot(frame["step"], gap, label=path_type, linewidth=1.25, color=colors.get(path_type))
        ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.6)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Synthetic evaluation step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    main()
