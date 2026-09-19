#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
SRC="$ROOT/source"
cd "$SRC"
for d in e2_results online_results variance_results variance_analysis figures tables; do
  [ ! -e "$ROOT/$d" ] || { echo "refusing: $ROOT/$d exists" >&2; exit 2; }
done
python run_e2_hybrid_v5.py --output "$ROOT/e2_results" --repetitions 600 --trials 400 --max-M 64 --workers 4
python analyze_e2_hybrid_v5.py
python run_online_hybrid_v5.py --output "$ROOT/online_results" --workers 4
python analyze_online_hybrid_v5.py
python run_variance_decomposition.py --output-root "$ROOT/variance_results" --workers 12 --replications 64 --experiments A B
python analyze_variance_decomposition.py --input-root "$ROOT/variance_results" --output-root "$ROOT/variance_analysis"
