from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvp_cpt_pg.config import DateWindow, ExperimentConfig
from mvp_cpt_pg.market_data import MarketDatasetBuilder, TushareDataClient
from mvp_cpt_pg.progress import progress


DEFAULT_START_DATE = "20230101"
DEFAULT_END_DATE = "20260531"
DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,dv_ratio,total_mv,circ_mv"
)
DEFAULT_STATUS_PATH = ROOT / "artifacts" / "inputs" / "prefetch_market_cache_status.csv"
DEFAULT_MARKET_CONTEXT_PATH = ROOT / "artifacts" / "inputs" / "market_context_20250515_20260512.csv"
def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch MVP market raw cache for a Tushare date window")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date in YYYYMMDD format")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="End date in YYYYMMDD format")
    parser.add_argument("--sleep-seconds", type=float, default=0.8, help="Sleep between Tushare daily and daily_basic requests")
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH, help="Output CSV path for cache status")
    parser.add_argument(
        "--market-context-path",
        type=Path,
        default=DEFAULT_MARKET_CONTEXT_PATH,
        help="Output CSV path for daily market context",
    )
    parser.add_argument(
        "--skip-market-context",
        action="store_true",
        help="Only prefetch raw Tushare cache and skip the 240-day market context file",
    )
    args = parser.parse_args()

    config = replace(
        ExperimentConfig(),
        prewarm=DateWindow(start=args.start_date, end=args.start_date),
        evaluation=DateWindow(start=args.start_date, end=args.end_date),
    )
    builder = MarketDatasetBuilder(config)
    status_rows: list[dict[str, Any]] = []

    trade_dates = prefetch_trade_calendar(
        builder.tushare,
        start_date=args.start_date,
        end_date=args.end_date,
        status_rows=status_rows,
    )
    stock_basic = prefetch_stock_basic(builder.tushare, status_rows)
    universe_codes = []
    if "ts_code" in stock_basic.columns:
        universe_codes = stock_basic["ts_code"].astype(str).tolist()

    prefetch_daily_series(builder.tushare, trade_dates, args.sleep_seconds, status_rows)
    prefetch_daily_basic_series(builder.tushare, trade_dates, args.sleep_seconds, status_rows)
    market_context = pd.DataFrame()
    if not args.skip_market_context:
        market_context = build_market_context(builder, config, trade_dates, universe_codes)

    status_path = args.status_path.resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_rows).to_csv(status_path, index=False, encoding="utf-8-sig")
    market_context_path = args.market_context_path.resolve()
    if not args.skip_market_context:
        market_context_path.parent.mkdir(parents=True, exist_ok=True)
        market_context.to_csv(market_context_path, index=False, encoding="utf-8-sig")

    daily_basic_empty_dates = [
        row["key"]
        for row in status_rows
        if row["dataset"] == "daily_basic" and row["success"] == 1 and row["rows"] == 0
    ]
    print(f"trade_dates={len(trade_dates)}")
    print(f"daily_basic_empty_dates={len(daily_basic_empty_dates)}")
    print(f"status_csv={status_path}")
    if args.skip_market_context:
        print("market_context_csv=skipped")
    else:
        print(f"market_context_csv={market_context_path}")


def prefetch_trade_calendar(
    tushare: TushareDataClient,
    *,
    start_date: str,
    end_date: str,
    status_rows: list[dict[str, Any]],
) -> list[str]:
    payload = {"start_date": start_date, "end_date": end_date}
    record = tushare.cache._record("trade_cal", payload)
    cached_before = record.data_path.exists()
    started_at = time.perf_counter()
    trade_dates = tushare.trade_calendar(start_date, end_date)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    status_rows.append(
        build_status_row(
            dataset="trade_cal",
            namespace="trade_cal",
            key=start_date,
            rows=len(trade_dates),
            success=1,
            cached_before=cached_before,
            cache_data_path=record.data_path,
            cache_meta_path=record.meta_path,
            payload=payload,
            elapsed_seconds=elapsed_seconds,
            note="open_trade_dates",
        )
    )
    return trade_dates


def prefetch_stock_basic(
    tushare: TushareDataClient,
    status_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    payload = {"fields": "ts_code,symbol,name,area,industry,market,list_date"}
    record = tushare.cache._record("stock_basic", payload)
    cached_before = record.data_path.exists()
    started_at = time.perf_counter()
    frame = tushare.stock_basic()
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    status_rows.append(
        build_status_row(
            dataset="stock_basic",
            namespace="stock_basic",
            key="all_listed",
            rows=len(frame),
            success=1,
            cached_before=cached_before,
            cache_data_path=record.data_path,
            cache_meta_path=record.meta_path,
            payload=payload,
            elapsed_seconds=elapsed_seconds,
            note="",
        )
    )
    return frame


def prefetch_daily_series(
    tushare: TushareDataClient,
    trade_dates: list[str],
    sleep_seconds: float,
    status_rows: list[dict[str, Any]],
) -> None:
    total = len(trade_dates)
    for index, trade_date in enumerate(progress(trade_dates, desc="tushare daily", total=total)):
        payload = {"trade_date": trade_date}
        record = tushare.cache._record("daily", payload)
        cached_before = record.data_path.exists()
        started_at = time.perf_counter()
        frame = tushare.daily_by_trade_date(trade_date)
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        status_rows.append(
            build_status_row(
                dataset="daily",
                namespace="daily",
                key=trade_date,
                rows=len(frame),
                success=1,
                cached_before=cached_before,
                cache_data_path=record.data_path,
                cache_meta_path=record.meta_path,
                payload=payload,
                elapsed_seconds=elapsed_seconds,
                note=window_label(trade_date),
            )
        )
        maybe_sleep(index=index, total=total, sleep_seconds=sleep_seconds)


def prefetch_daily_basic_series(
    tushare: TushareDataClient,
    trade_dates: list[str],
    sleep_seconds: float,
    status_rows: list[dict[str, Any]],
) -> None:
    total = len(trade_dates)
    for index, trade_date in enumerate(progress(trade_dates, desc="tushare daily_basic", total=total)):
        payload = {
            "trade_date": trade_date,
            "fields": DAILY_BASIC_FIELDS,
        }
        record = tushare.cache._record("daily_basic", payload)
        cached_before = record.data_path.exists()
        started_at = time.perf_counter()
        frame = tushare.daily_basic_by_trade_date(trade_date)
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        note = window_label(trade_date)
        if frame.empty:
            note = f"{note};empty_return"
        status_rows.append(
            build_status_row(
                dataset="daily_basic",
                namespace="daily_basic",
                key=trade_date,
                rows=len(frame),
                success=1,
                cached_before=cached_before,
                cache_data_path=record.data_path,
                cache_meta_path=record.meta_path,
                payload=payload,
                elapsed_seconds=elapsed_seconds,
                note=note,
            )
        )
        maybe_sleep(index=index, total=total, sleep_seconds=sleep_seconds)


def build_status_row(
    *,
    dataset: str,
    namespace: str,
    key: str,
    rows: int,
    success: int,
    cached_before: bool,
    cache_data_path: Path | None,
    cache_meta_path: Path | None,
    payload: dict[str, Any],
    elapsed_seconds: float,
    note: str,
    error: str = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "namespace": namespace,
        "key": key,
        "rows": rows,
        "success": success,
        "cached_before": int(cached_before),
        "cache_data_path": "" if cache_data_path is None else str(cache_data_path),
        "cache_meta_path": "" if cache_meta_path is None else str(cache_meta_path),
        "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        "elapsed_seconds": elapsed_seconds,
        "note": note,
        "error": error,
    }


def build_market_context(
    builder: MarketDatasetBuilder,
    config: ExperimentConfig,
    trade_dates: list[str],
    universe_codes: list[str],
) -> pd.DataFrame:
    evaluation_dates = [date for date in trade_dates if config.evaluation.start <= date <= config.evaluation.end]
    if len(evaluation_dates) != 240:
        raise RuntimeError(f"Evaluation window has {len(evaluation_dates)} trade dates, expected 240")
    universe_set = set(universe_codes)
    rows = []
    for trade_date in progress(evaluation_dates, desc="market context", total=len(evaluation_dates)):
        daily = builder.tushare.daily_by_trade_date(trade_date)
        daily_basic = builder.tushare.daily_basic_by_trade_date(trade_date)
        if universe_set and "ts_code" in daily.columns:
            daily = daily[daily["ts_code"].astype(str).isin(universe_set)].copy()
        if universe_set and "ts_code" in daily_basic.columns:
            daily_basic = daily_basic[daily_basic["ts_code"].astype(str).isin(universe_set)].copy()
        rows.append(market_context_row(trade_date, daily, daily_basic))
    return pd.DataFrame(rows).fillna(0)


def market_context_row(trade_date: str, daily: pd.DataFrame, daily_basic: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        raise RuntimeError(f"daily is empty for {trade_date}")
    pct = pd.to_numeric(daily["pct_chg"], errors="coerce").dropna() / 100.0
    amount = pd.to_numeric(daily["amount"], errors="coerce").fillna(0.0)
    output: dict[str, Any] = {
        "trade_date": trade_date,
        "market_ret_mean": float(pct.mean()),
        "market_ret_median": float(pct.median()),
        "market_ret_std": float(pct.std(ddof=0)),
        "advancing_ratio": float((pct > 0).mean()),
        "declining_ratio": float((pct < 0).mean()),
        "market_amount_sum": float(amount.sum()),
        "market_amount_mean": float(amount.mean()),
    }
    if daily_basic.empty:
        output.update(
            {
                "turnover_rate_mean": 0.0,
                "volume_ratio_mean": 0.0,
                "pe_median": 0.0,
                "pb_median": 0.0,
                "circ_mv_sum": 0.0,
            }
        )
        return output
    output.update(
        {
            "turnover_rate_mean": numeric_mean(daily_basic, "turnover_rate"),
            "volume_ratio_mean": numeric_mean(daily_basic, "volume_ratio"),
            "pe_median": numeric_median(daily_basic, "pe"),
            "pb_median": numeric_median(daily_basic, "pb"),
            "circ_mv_sum": numeric_sum(daily_basic, "circ_mv"),
        }
    )
    return output


def numeric_mean(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def numeric_median(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").median())


def numeric_sum(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def maybe_sleep(*, index: int, total: int, sleep_seconds: float) -> None:
    if sleep_seconds > 0 and index + 1 < total:
        time.sleep(sleep_seconds)


def window_label(trade_date: str) -> str:
    return "requested_window"


if __name__ == "__main__":
    main()
