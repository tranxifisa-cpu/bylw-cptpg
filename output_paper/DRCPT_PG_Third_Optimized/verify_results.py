"""Reconstruct the reported numerical summaries from released replication records."""
from pathlib import Path
import csv
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent

def close(actual, expected):
    if not np.isclose(actual, expected, rtol=2e-12, atol=2e-14):
        raise AssertionError((actual, expected))

def verify():
    data = ROOT / "results"
    result = json.loads((data / "full_results.json").read_text())
    levels = np.genfromtxt(data / "full_level_replications.csv", delimiter=",", names=True)
    pool = np.genfromtxt(data / "full_capped_replications.csv", delimiter=",", names=True)
    gradient = result["exact_gradient_quadrature"]
    assert len(levels) == 9600 and len(pool) == 8192
    assert np.all(np.isfinite(pool["Z_db"]))
    for row in result["levels"]:
        records = levels[levels["level"] == row["level"]]
        assert len(records) == 1600 and np.all(records["n"] == row["n"])
        close(records["T_fine"].mean() - gradient, row["empirical_bias"])
        close(records["T_fine"].std(ddof=1) / math.sqrt(len(records)), row["bias_se"])
        squares = records["Delta"] ** 2
        close(squares.mean(), row["delta_second_moment"])
        close(squares.std(ddof=1) / math.sqrt(len(records)), row["delta_second_moment_se"])
        close(result["analytic_bias_coefficient"] / row["n"], row["analytic_bias"])
    for row in result["batches"]:
        size = row["M"]
        means = pool["Z_db"].reshape(-1, size).mean(axis=1)
        costs = pool["raw_trajectory_cost"].reshape(-1, size).sum(axis=1)
        losses = (means - gradient) ** 2
        assert len(means) == row["groups"]
        close(losses.mean(), row["mse"])
        close(losses.std(ddof=1) / math.sqrt(len(means)), row["mse_se"])
        close(costs.mean(), row["average_trajectory_cost"])
    with (data / "full_tracking.csv").open() as stream:
        records = list(csv.DictReader(stream))
    tracking = result["tracking"]
    assert len(records) == tracking["K"]
    close(np.mean([float(r["Q_quadrature"]) ** 2 for r in records]), tracking["average_squared_Q"])
    close(np.mean([(float(r["estimated_gradient"]) - float(r["true_gradient_quadrature"])) ** 2
                   for r in records]), tracking["average_squared_gradient_error"])
    assert sum(int(r["raw_cost"]) for r in records) == tracking["total_raw_trajectories"]
    for field, key in [("next_theta", "final_theta"), ("next_X", "final_wealth"),
                       ("next_r", "final_reference")]:
        close(float(records[-1][field]), tracking[key])
    return {"level_replications": len(levels), "capped_replications": len(pool),
            "episodes": len(records), "summaries_reconstructed": True}

if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
