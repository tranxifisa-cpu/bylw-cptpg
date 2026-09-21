from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvp_cpt_pg.raw_cache import DataFrameCache


FIELDS = "ts_code,ann_date,end_date,roe,roa,roic"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hs300_snapshot(cache_root: Path, selection_date: str) -> tuple[str, list[str]]:
    frames = []
    for meta_path in sorted((cache_root / "index_weight").glob("*.json")):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("index_code") != "000300.SH":
            continue
        data_path = meta_path.with_suffix(".pkl")
        if data_path.exists():
            frames.append(pd.read_pickle(data_path))
    if not frames:
        raise FileNotFoundError("No cached HS300 index weights")
    weights = pd.concat(frames, ignore_index=True)
    weights["trade_date"] = weights["trade_date"].astype(str)
    eligible = weights[weights.trade_date <= selection_date]
    if eligible.empty:
        raise ValueError(f"No HS300 snapshot on or before {selection_date}")
    snapshot = str(eligible.trade_date.max())
    codes = sorted(eligible.loc[eligible.trade_date == snapshot, "con_code"].astype(str).unique())
    if len(codes) != 300:
        raise ValueError(f"HS300 snapshot {snapshot} has {len(codes)} constituents, expected 300")
    return snapshot, codes


def fetch_with_retry(pro, code: str, start_date: str, end_date: str,
                     attempts: int, base_sleep: float) -> pd.DataFrame:
    for attempt in range(attempts):
        try:
            return pro.fina_indicator(
                ts_code=code,
                start_date=start_date,
                end_date=end_date,
                fields=FIELDS,
            )
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(base_sleep * (2 ** attempt))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "artifacts/cache/raw/tushare")
    parser.add_argument("--selection-date", default="20221230")
    parser.add_argument("--start-date", default="20200101")
    parser.add_argument("--end-date", default="20260529")
    parser.add_argument("--sleep-seconds", type=float, default=0.8)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "artifacts/inputs/paper_hs300_fundamentals.json")
    args = parser.parse_args()

    token = os.getenv("TUSHARE_TOKEN") or os.getenv("Tushare_Token")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    snapshot, codes = hs300_snapshot(args.cache, args.selection_date)
    pro = ts.pro_api(token)
    cache = DataFrameCache(args.cache)
    completed = 0
    total_rows = 0
    records = []
    for index, code in enumerate(codes, 1):
        payload = {
            "ts_code": code,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "fields": FIELDS,
        }
        record = cache._record("fina_indicator", payload)
        cached = record.data_path.exists()
        frame = cache.get_or_fetch(
            "fina_indicator",
            payload,
            lambda code=code: fetch_with_retry(
                pro, code, args.start_date, args.end_date,
                args.attempts, args.sleep_seconds,
            ),
        )
        if frame.empty:
            raise ValueError(f"No financial indicators returned for HS300 constituent {code}")
        required = {"ts_code", "ann_date", "roe", "roa", "roic"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Incomplete financial-indicator fields for {code}")
        if set(frame["ts_code"].astype(str)) != {code}:
            raise ValueError(f"Financial-indicator cache contains the wrong stock for {code}")
        completed += 1
        total_rows += len(frame)
        ann_dates = frame["ann_date"].dropna().astype(str).str[:8]
        records.append({
            "ts_code": code,
            "rows": len(frame),
            "announcement_start": str(ann_dates.min()),
            "announcement_end": str(ann_dates.max()),
            "data_sha256": file_hash(record.data_path),
        })
        print(f"HS300 fundamentals {index:03d}/300 {code}: {len(frame)} rows"
              f" ({'cached' if cached else 'fetched'})", flush=True)
        if not cached:
            time.sleep(args.sleep_seconds)
    manifest = {
        "status": "complete",
        "source": "Tushare fina_indicator",
        "snapshot": snapshot,
        "constituents": len(codes),
        "completed": completed,
        "rows": total_rows,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "fields": FIELDS.split(","),
        "point_in_time_rule": "Only announcements with ann_date strictly before the trading date are used",
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
