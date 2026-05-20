from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_MVP = ROOT / "scripts" / "run_mvp.py"
LOG_ROOT = ROOT / "artifacts" / "results" / "grid_logs"


@dataclass(frozen=True)
class GridJob:
    name: str
    methods: tuple[str, ...]
    gamma0: float
    policy_noise_scale: float
    gamma_exponent: float
    eta_gain: float
    eta_loss: float
    evaluation_horizon: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPT-PG grid-search jobs in a bounded parallel queue")
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3"), required=True)
    parser.add_argument("--max-workers", type=int, default=2, help="Concurrent run_mvp processes")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--dry-run-days", type=int, default=30)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--preference-path", type=Path, default=None)
    parser.add_argument("--prewarm-start", default=None)
    parser.add_argument("--prewarm-end", default=None)
    parser.add_argument("--evaluation-start", default=None)
    parser.add_argument("--evaluation-end", default=None)
    parser.add_argument("--cpt-sample-base", type=int, default=256)
    parser.add_argument("--gradient-sample-base", type=int, default=32)
    parser.add_argument("--gamma0", type=float, default=0.3, help="Fixed gamma0 for stages 2 and 3")
    parser.add_argument("--policy-noise-scale", type=float, default=0.15, help="Fixed policy noise for stages 2 and 3")
    parser.add_argument("--gamma-exponent", type=float, default=0.51)
    parser.add_argument("--eta-gain", type=float, default=0.40, help="Fixed eta_plus for stages 1 and 3")
    parser.add_argument("--eta-loss", type=float, default=0.30, help="Fixed eta_minus for stages 1 and 3")
    parser.add_argument("--evaluation-horizon", type=int, default=15, help="Fixed h for stages 1 and 2")
    args = parser.parse_args()

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")

    jobs = build_jobs(args)
    run_id = time.strftime("grid_%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=False)

    manifest_path = log_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "status",
                "returncode",
                "result_dir",
                "seconds",
                "log_path",
                "methods",
                "gamma0",
                "policy_noise_scale",
                "gamma_exponent",
                "eta_gain",
                "eta_loss",
                "evaluation_horizon",
            ],
        )
        writer.writeheader()

    print(f"grid_log_dir: {log_dir}")
    print(f"job_count: {len(jobs)}")
    print(f"max_workers: {args.max_workers}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_job, args, job, log_dir, index): job for index, job in enumerate(jobs, start=1)}
        for future in as_completed(futures):
            result = future.result()
            append_manifest(manifest_path, result)
            print(
                f"{result['status']}: {result['name']} "
                f"seconds={result['seconds']:.1f} result_dir={result['result_dir']}"
            )


def build_jobs(args: argparse.Namespace) -> list[GridJob]:
    if args.stage == "stage1":
        gamma_values = (0.1, 0.3, 1.0, 3.0)
        sigma_values = (0.10, 0.15, 0.25, 0.40)
        return [
            GridJob(
                name=f"stage1_g{gamma0}_s{sigma}",
                methods=("dynamic_cpt_pg", "static_cpt_pg"),
                gamma0=gamma0,
                policy_noise_scale=sigma,
                gamma_exponent=args.gamma_exponent,
                eta_gain=args.eta_gain,
                eta_loss=args.eta_loss,
                evaluation_horizon=args.evaluation_horizon,
            )
            for gamma0 in gamma_values
            for sigma in sigma_values
        ]
    if args.stage == "stage2":
        eta_pairs = (
            (0.20, 0.05),
            (0.20, 0.10),
            (0.40, 0.05),
            (0.40, 0.10),
            (0.40, 0.20),
            (0.60, 0.05),
            (0.60, 0.10),
            (0.60, 0.20),
            (0.60, 0.40),
            (0.10, 0.05),
        )
        return [
            GridJob(
                name=f"stage2_ep{eta_gain}_em{eta_loss}",
                methods=("dynamic_cpt_pg", "static_cpt_pg", "dynamic_cpt_pg_frozen_pref", "static_ref_dynamic_pref_cpt_pg"),
                gamma0=args.gamma0,
                policy_noise_scale=args.policy_noise_scale,
                gamma_exponent=args.gamma_exponent,
                eta_gain=eta_gain,
                eta_loss=eta_loss,
                evaluation_horizon=args.evaluation_horizon,
            )
            for eta_gain, eta_loss in eta_pairs
        ]
    h_values = (5, 10, 15, 20, 30)
    return [
        GridJob(
            name=f"stage3_h{h}",
            methods=("dynamic_cpt_pg", "static_cpt_pg", "dynamic_cpt_pg_frozen_pref", "static_ref_dynamic_pref_cpt_pg"),
            gamma0=args.gamma0,
            policy_noise_scale=args.policy_noise_scale,
            gamma_exponent=args.gamma_exponent,
            eta_gain=args.eta_gain,
            eta_loss=args.eta_loss,
            evaluation_horizon=h,
        )
        for h in h_values
    ]


def run_job(args: argparse.Namespace, job: GridJob, log_dir: Path, index: int) -> dict[str, object]:
    command = [
        sys.executable,
        str(RUN_MVP),
        "--methods",
        *job.methods,
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--dry-run-days",
        str(args.dry_run_days),
        "--initial-capital",
        str(args.initial_capital),
        "--fixed-sample-counts",
        "--cpt-sample-base",
        str(args.cpt_sample_base),
        "--gradient-sample-base",
        str(args.gradient_sample_base),
        "--gamma0",
        str(job.gamma0),
        "--gamma-exponent",
        str(job.gamma_exponent),
        "--policy-noise-scale",
        str(job.policy_noise_scale),
        "--eta-gain",
        str(job.eta_gain),
        "--eta-loss",
        str(job.eta_loss),
        "--evaluation-horizon",
        str(job.evaluation_horizon),
    ]
    append_optional(command, "--preference-path", args.preference_path)
    append_optional(command, "--prewarm-start", args.prewarm_start)
    append_optional(command, "--prewarm-end", args.prewarm_end)
    append_optional(command, "--evaluation-start", args.evaluation_start)
    append_optional(command, "--evaluation-end", args.evaluation_end)

    started = time.perf_counter()
    log_path = log_dir / f"{index:03d}_{job.name}.log"
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    seconds = time.perf_counter() - started
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    result_dir = parse_result_dir(completed.stdout)
    return {
        "name": job.name,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "result_dir": result_dir,
        "seconds": seconds,
        "log_path": str(log_path),
        "methods": " ".join(job.methods),
        "gamma0": job.gamma0,
        "policy_noise_scale": job.policy_noise_scale,
        "gamma_exponent": job.gamma_exponent,
        "eta_gain": job.eta_gain,
        "eta_loss": job.eta_loss,
        "evaluation_horizon": job.evaluation_horizon,
    }


def append_optional(command: list[str], option: str, value: object | None) -> None:
    if value is not None:
        command.extend([option, str(value)])


def parse_result_dir(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("result_dir:"):
            return line.split(":", 1)[1].strip()
    return ""


def append_manifest(manifest_path: Path, row: dict[str, object]) -> None:
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


if __name__ == "__main__":
    main()
