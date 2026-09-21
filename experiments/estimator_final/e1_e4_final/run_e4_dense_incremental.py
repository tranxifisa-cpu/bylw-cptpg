from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from run_e4_slow import E4_SEED_BASE, L_DIAG, one

# Dense E4 design frozen after the E3/E4 interpretation audit.
REFERENCE_GRID = tuple(round(0.05 * i, 10) for i in range(9))  # 0.00,...,0.40
HORIZON_GRID = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
GAMMA_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)
VARTTHETA_GRID = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06)


def admissible(gamma: float, vartheta: float) -> bool:
    return float(gamma) * L_DIAG <= float(vartheta) + 1e-15


def optimizer_pairs() -> list[tuple[float, float]]:
    return [(g, v) for g in GAMMA_GRID for v in VARTTHETA_GRID if admissible(g, v)]


def _norm_key2(value) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return round(float(value), 12)


def row_key(panel, key1, key2, seed) -> tuple[str, float, float | None, int]:
    return (str(panel), round(float(key1), 12), _norm_key2(key2), int(seed))


def target_jobs(seeds: int) -> list[tuple[str, float, float | None, int]]:
    jobs: list[tuple[str, float, float | None, int]] = []
    for eta_gain in REFERENCE_GRID:
        for eta_loss in REFERENCE_GRID:
            for s in range(seeds):
                jobs.append(("reference", eta_gain, eta_loss, E4_SEED_BASE + s))
    for horizon in HORIZON_GRID:
        for s in range(seeds):
            jobs.append(("horizon", float(horizon), None, E4_SEED_BASE + s))
    for gamma, vartheta in optimizer_pairs():
        for s in range(seeds):
            jobs.append(("optimizer", gamma, vartheta, E4_SEED_BASE + s))
    return jobs


def load_rows(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    out = pd.read_csv(path)
    needed = {"panel", "key1", "key2", "seed", "online_dlr", "frozen_dlr"}
    missing = sorted(needed.difference(out.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return out


def dedupe_to_target(frames: list[pd.DataFrame], targets: set[tuple]) -> pd.DataFrame:
    chosen: dict[tuple, dict] = {}
    # Later frames override earlier ones. This lets a checkpoint override reused legacy rows.
    for frame in frames:
        if frame.empty:
            continue
        for row in frame.to_dict("records"):
            k = row_key(row["panel"], row["key1"], row.get("key2"), row["seed"])
            if k in targets:
                chosen[k] = row
    if not chosen:
        return pd.DataFrame()
    out = pd.DataFrame(chosen.values())
    out["dlr_ratio"] = out["online_dlr"] / out["frozen_dlr"]
    return out.sort_values(["panel", "key1", "key2", "seed"], na_position="last").reset_index(drop=True)


def checkpoint(path: Path, rows: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    rows.to_csv(tmp, index=False)
    tmp.replace(path)


def protocol(seeds: int) -> dict:
    pairs = optimizer_pairs()
    return {
        "seed_base": E4_SEED_BASE,
        "seeds": int(seeds),
        "baseline": {"gamma": 0.08, "vartheta": 0.05, "eta_gain": 0.20, "eta_loss": 0.05, "horizon": 5},
        "reference_grid": list(REFERENCE_GRID),
        "horizon_grid": list(HORIZON_GRID),
        "optimizer_grid": {
            "gamma": list(GAMMA_GRID),
            "vartheta": list(VARTTHETA_GRID),
            "L_diag": L_DIAG,
            "admissible": pairs,
        },
        "primary_display_quantity": "post-change DLR ratio = DLR_online / DLR_frozen within the same configuration",
        "interpretation": "configuration-matched closed-loop robustness effect; values below 1 mean lower post-change DLR under continued online updating",
        "environment": "same continuous slow-variation setup as E3; state gain tied to elapsed market time; total market time 1500 days",
        "role": "robustness only; E4 does not select the default configuration",
        "target_seed_level_jobs": 81 * seeds + len(HORIZON_GRID) * seeds + len(pairs) * seeds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume-capable dense E4 robustness sweep.")
    ap.add_argument("--output", type=Path, required=True, help="Directory for protocol/checkpoints/final seed_metrics.csv")
    ap.add_argument("--reuse", type=Path, default=None, help="Existing sparse E4 seed_metrics.csv to reuse")
    ap.add_argument("--resume-from", type=Path, default=None, help="Optional additional partial CSV from a previous dense run")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--plan-only", action="store_true", help="Print reuse/missing counts without running simulations")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    targets_list = target_jobs(args.seeds)
    targets = set(targets_list)

    existing_final = args.output / "seed_metrics.csv"
    existing_partial = args.output / "seed_metrics_partial.csv"
    frames = [load_rows(args.reuse), load_rows(args.resume_from), load_rows(existing_partial), load_rows(existing_final)]
    rows = dedupe_to_target(frames, targets)
    have = set()
    if not rows.empty:
        have = {row_key(r.panel, r.key1, r.key2, r.seed) for r in rows.itertuples(index=False)}
    missing = [j for j in targets_list if row_key(*j) not in have]

    proto = protocol(args.seeds)
    proto.update({"reused_or_resumed_jobs": len(have), "missing_jobs_at_start": len(missing)})
    (args.output / "protocol_dense.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")

    print(json.dumps({
        "target_jobs": len(targets_list),
        "available_jobs": len(have),
        "missing_jobs": len(missing),
        "reference_configs": len(REFERENCE_GRID) ** 2,
        "horizon_configs": len(HORIZON_GRID),
        "optimizer_configs": len(optimizer_pairs()),
    }, indent=2), flush=True)

    if args.plan_only:
        return
    if not missing:
        checkpoint(existing_final, rows)
        (args.output / "completion_dense.json").write_text(
            json.dumps({"status": "complete", "runtime_seconds": 0.0, "jobs": len(rows)}, indent=2), encoding="utf-8"
        )
        return

    start = time.time()
    new_records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futures = {ex.submit(one, j): j for j in missing}
        for idx, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            result["dlr_ratio"] = float(result["online_dlr"] / result["frozen_dlr"])
            new_records.append(result)
            print("E4-dense", result["panel"], result["key1"], result["key2"], result["seed"], "done", flush=True)
            if idx % max(args.checkpoint_every, 1) == 0:
                new_frame = pd.DataFrame(new_records)
                merged = dedupe_to_target([rows, new_frame], targets)
                checkpoint(existing_partial, merged)

    merged = dedupe_to_target([rows, pd.DataFrame(new_records)], targets)
    checkpoint(existing_partial, merged)
    if len(merged) != len(targets_list):
        raise RuntimeError(f"dense E4 incomplete after run: {len(merged)}/{len(targets_list)} rows")
    checkpoint(existing_final, merged)
    (args.output / "completion_dense.json").write_text(
        json.dumps({"status": "complete", "runtime_seconds": time.time() - start, "jobs": len(merged)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
