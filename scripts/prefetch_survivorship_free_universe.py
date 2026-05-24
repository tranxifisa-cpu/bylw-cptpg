from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvp_cpt_pg.config import DateWindow, ExperimentConfig
from mvp_cpt_pg.market_data import MarketDatasetBuilder
from mvp_cpt_pg.progress import progress
from mvp_cpt_pg.utils import ensure_dir


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "cache" / "universe_by_date"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch a survivorship-free daily tradable A-share universe cache",
    )
    parser.add_argument("--prewarm-start", default="20250414", help="Prewarm start date, YYYYMMDD")
    parser.add_argument("--prewarm-end", default="20250514", help="Prewarm end date, YYYYMMDD")
    parser.add_argument("--evaluation-start", default="20250515", help="Evaluation start date, YYYYMMDD")
    parser.add_argument("--evaluation-end", default="20260512", help="Evaluation end date, YYYYMMDD")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=120,
        help="Calendar-day lookback before prewarm-start used by the market panel",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.35,
        help="Sleep between uncached Tushare daily requests to respect rate limits",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = replace(
        ExperimentConfig(),
        prewarm=DateWindow(start=args.prewarm_start, end=args.prewarm_end),
        evaluation=DateWindow(start=args.evaluation_start, end=args.evaluation_end),
    )
    builder = MarketDatasetBuilder(config)
    full_start = (pd.Timestamp(config.prewarm.start) - timedelta(days=args.lookback_days)).strftime("%Y%m%d")
    trade_dates = builder.tushare.trade_calendar(full_start, config.evaluation.end)
    stock_basic = builder.tushare.stock_basic_all_status()
    eligible_basic = eligible_stock_basic(stock_basic, config)

    rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    total = len(trade_dates)
    for index, trade_date in enumerate(progress(trade_dates, desc="daily tradable universe", total=total)):
        payload = {"trade_date": trade_date}
        record = builder.tushare.cache._record("daily", payload)
        cached_before = record.data_path.exists()
        started_at = time.perf_counter()
        daily = builder.tushare.daily_by_trade_date(trade_date)
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        daily_universe = tradable_universe_for_date(trade_date, daily, eligible_basic)
        rows.append(daily_universe)
        status_rows.append(
            {
                "trade_date": trade_date,
                "rows": int(len(daily_universe)),
                "daily_cached_before": int(cached_before),
                "cache_data_path": str(record.data_path),
                "elapsed_seconds": elapsed_seconds,
            }
        )
        if not cached_before and args.sleep_seconds > 0 and index + 1 < total:
            time.sleep(args.sleep_seconds)

    output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=output_columns())
    output = output.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    output_dir = ensure_dir(args.output_dir)
    output_path = output_dir / f"survivorship_free_universe_{full_start}_{config.evaluation.end}.csv"
    status_path = output_dir / f"survivorship_free_universe_status_{full_start}_{config.evaluation.end}.csv"
    meta_path = output_dir / f"survivorship_free_universe_{full_start}_{config.evaluation.end}.json"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(status_rows).to_csv(status_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(
        json.dumps(
            {
                "prewarm": config.prewarm.__dict__,
                "evaluation": config.evaluation.__dict__,
                "full_start": full_start,
                "full_end": config.evaluation.end,
                "trade_dates": len(trade_dates),
                "stock_basic_all_status_rows": int(len(stock_basic)),
                "eligible_non_financial_rows": int(len(eligible_basic)),
                "universe_rows": int(len(output)),
                "unique_stocks": int(output["ts_code"].nunique()) if not output.empty else 0,
                "filters": [
                    "list_date <= trade_date",
                    "delist_date is blank or delist_date > trade_date",
                    "industry not in finance_industries",
                    "daily row exists on trade_date",
                    "amount > 0 and vol > 0 when these columns are available",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"trade_dates={len(trade_dates)}")
    print(f"eligible_non_financial_stocks={len(eligible_basic)}")
    print(f"universe_rows={len(output)}")
    print(f"unique_stocks={output['ts_code'].nunique() if not output.empty else 0}")
    print(f"universe_by_date_path={output_path}")
    print(f"status_path={status_path}")
    print(f"meta_path={meta_path}")


def eligible_stock_basic(stock_basic: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    required = {"ts_code", "name", "industry", "list_date"}
    missing = required.difference(stock_basic.columns)
    if missing:
        raise RuntimeError(f"stock_basic_all_status is missing required columns: {sorted(missing)}")
    frame = stock_basic.copy()
    if "delist_date" not in frame.columns:
        frame["delist_date"] = ""
    if "list_status" not in frame.columns:
        frame["list_status"] = ""
    for column in output_columns():
        if column not in frame.columns and column != "trade_date":
            frame[column] = ""
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["industry"] = frame["industry"].fillna("未知行业")
    frame["list_date"] = frame["list_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    frame["delist_date"] = frame["delist_date"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    frame = frame[~frame["industry"].isin(config.finance_industries)].copy()
    return frame.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)


def tradable_universe_for_date(trade_date: str, daily: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=output_columns())
    daily_codes = daily.copy()
    daily_codes["ts_code"] = daily_codes["ts_code"].astype(str)
    if "amount" in daily_codes.columns:
        daily_codes["amount"] = pd.to_numeric(daily_codes["amount"], errors="coerce").fillna(0.0)
        daily_codes = daily_codes[daily_codes["amount"] > 0.0].copy()
    if "vol" in daily_codes.columns:
        daily_codes["vol"] = pd.to_numeric(daily_codes["vol"], errors="coerce").fillna(0.0)
        daily_codes = daily_codes[daily_codes["vol"] > 0.0].copy()
    frame = stock_basic.merge(daily_codes[["ts_code"]].drop_duplicates(), on="ts_code", how="inner")
    frame = frame[frame["list_date"] <= trade_date].copy()
    delist = frame["delist_date"].fillna("").astype(str)
    frame = frame[(delist.eq("")) | (delist.eq("nan")) | (delist > trade_date)].copy()
    frame["trade_date"] = trade_date
    return frame.reindex(columns=output_columns()).copy()


def output_columns() -> list[str]:
    return ["trade_date", "ts_code", "symbol", "name", "area", "industry", "market", "list_date", "delist_date", "list_status"]


if __name__ == "__main__":
    main()
