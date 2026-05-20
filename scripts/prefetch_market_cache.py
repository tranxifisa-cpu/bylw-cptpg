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
from mvp_cpt_pg.market_data import AkshareStockInfoClient, MarketDatasetBuilder, TushareDataClient
from mvp_cpt_pg.progress import progress


PREWARM_WINDOW = DateWindow(start="20250414", end="20250514")
EVAL_WINDOW = DateWindow(start="20250515", end="20260512")
DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,dv_ratio,total_mv,circ_mv"
)
DEFAULT_STATUS_PATH = ROOT / "artifacts" / "inputs" / "prefetch_market_cache_status.csv"
DEFAULT_MARKET_CONTEXT_PATH = ROOT / "artifacts" / "inputs" / "market_context_20250515_20260512.csv"
EXTRA_AKSHARE_SOURCES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("stock_board_industry_name_em", {}),
    ("stock_board_industry_summary_ths", {}),
    ("stock_fund_flow_industry", {"symbol": "即时"}),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch MVP market raw cache for the fixed prewarm and evaluation windows")
    parser.add_argument("--sleep-seconds", type=float, default=0.35, help="Sleep between Tushare daily and daily_basic requests")
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH, help="Output CSV path for cache status")
    parser.add_argument(
        "--market-context-path",
        type=Path,
        default=DEFAULT_MARKET_CONTEXT_PATH,
        help="Output CSV path for daily market and news context",
    )
    args = parser.parse_args()

    config = replace(
        ExperimentConfig(),
        prewarm=PREWARM_WINDOW,
        evaluation=EVAL_WINDOW,
    )
    builder = MarketDatasetBuilder(config)
    status_rows: list[dict[str, Any]] = []

    full_start = (pd.Timestamp(config.prewarm.start) - timedelta(days=120)).strftime("%Y%m%d")
    full_end = config.evaluation.end
    trade_dates = prefetch_trade_calendar(
        builder.tushare,
        start_date=full_start,
        end_date=full_end,
        status_rows=status_rows,
    )
    stock_basic = prefetch_stock_basic(builder.tushare, status_rows)
    universe_codes = []
    if "ts_code" in stock_basic.columns:
        universe_codes = stock_basic["ts_code"].astype(str).tolist()

    prefetch_daily_series(builder.tushare, trade_dates, args.sleep_seconds, status_rows)
    prefetch_daily_basic_series(builder.tushare, trade_dates, args.sleep_seconds, status_rows)
    prefetch_akshare_news(builder.akshare, config.news_sources, status_rows)
    prefetch_akshare_static(builder.akshare, config.static_stock_info_sources, universe_codes, status_rows)
    prefetch_extra_akshare(builder.akshare, status_rows)
    market_context = build_market_context(builder, config, trade_dates, universe_codes)

    status_path = args.status_path.resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_rows).to_csv(status_path, index=False, encoding="utf-8-sig")
    market_context_path = args.market_context_path.resolve()
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


def prefetch_akshare_news(
    akshare: AkshareStockInfoClient,
    news_sources: tuple[str, ...],
    status_rows: list[dict[str, Any]],
) -> None:
    cached_before_map = {
        source_name: akshare.cache._record(source_name, news_source_kwargs(source_name)).data_path.exists()
        for source_name in news_sources
    }
    started_at = time.perf_counter()
    _, source_status = akshare.fetch_news_sources()
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    for item in source_status:
        source_name = str(item["source"])
        payload = news_source_kwargs(source_name)
        record = akshare.cache._record(source_name, payload)
        status_rows.append(
            build_status_row(
                dataset="akshare_news",
                namespace=source_name,
                key=source_name,
                rows=int(item["rows"]),
                success=int(item["success"]),
                cached_before=cached_before_map.get(source_name, False),
                cache_data_path=record.data_path,
                cache_meta_path=record.meta_path,
                payload=payload,
                elapsed_seconds=elapsed_seconds,
                note="",
                error=str(item["error"]),
            )
        )


def prefetch_akshare_static(
    akshare: AkshareStockInfoClient,
    static_sources: tuple[str, ...],
    universe_codes: list[str],
    status_rows: list[dict[str, Any]],
) -> None:
    cached_before_map = {
        source_name: akshare.cache._record(source_name, static_source_kwargs(source_name)).data_path.exists()
        for source_name in static_sources
        if source_name != "stock_info_change_name"
    }
    started_at = time.perf_counter()
    _, source_status = akshare.fetch_static_sources(universe_codes)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    for item in source_status:
        source_name = str(item["source"])
        if source_name == "stock_info_change_name:per_symbol":
            status_rows.append(
                build_status_row(
                    dataset="akshare_static",
                    namespace="stock_info_change_name",
                    key=source_name,
                    rows=int(item["rows"]),
                    success=int(item["success"]),
                    cached_before=False,
                    cache_data_path=None,
                    cache_meta_path=None,
                    payload={},
                    elapsed_seconds=elapsed_seconds,
                    note="full_universe_skipped_by_existing_client",
                    error=str(item["error"]),
                )
            )
            continue
        payload = static_source_kwargs(source_name)
        record = akshare.cache._record(source_name, payload)
        status_rows.append(
            build_status_row(
                dataset="akshare_static",
                namespace=source_name,
                key=source_name,
                rows=int(item["rows"]),
                success=int(item["success"]),
                cached_before=cached_before_map.get(source_name, False),
                cache_data_path=record.data_path,
                cache_meta_path=record.meta_path,
                payload=payload,
                elapsed_seconds=elapsed_seconds,
                note="",
                error=str(item["error"]),
            )
        )


def prefetch_extra_akshare(akshare: AkshareStockInfoClient, status_rows: list[dict[str, Any]]) -> None:
    for source_name, kwargs in EXTRA_AKSHARE_SOURCES:
        payload = {"source_name": source_name, **kwargs}
        record = akshare.cache._record(source_name, payload)
        cached_before = record.data_path.exists()
        started_at = time.perf_counter()
        try:
            frame = akshare.fetch_source(source_name, **kwargs)
            status_rows.append(
                build_status_row(
                    dataset="akshare_industry",
                    namespace=source_name,
                    key=source_name,
                    rows=len(frame),
                    success=1,
                    cached_before=cached_before,
                    cache_data_path=record.data_path,
                    cache_meta_path=record.meta_path,
                    payload=payload,
                    elapsed_seconds=round(time.perf_counter() - started_at, 3),
                    note="industry_snapshot",
                    error="",
                )
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append(
                build_status_row(
                    dataset="akshare_industry",
                    namespace=source_name,
                    key=source_name,
                    rows=0,
                    success=0,
                    cached_before=cached_before,
                    cache_data_path=record.data_path,
                    cache_meta_path=record.meta_path,
                    payload=payload,
                    elapsed_seconds=round(time.perf_counter() - started_at, 3),
                    note="industry_snapshot",
                    error=repr(exc),
                )
            )


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
    context = pd.DataFrame(rows)
    news_daily, _ = builder._build_news_features(evaluation_dates, universe_codes)
    return context.merge(news_daily, on="trade_date", how="left").fillna(0)


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


def news_source_kwargs(source_name: str) -> dict[str, Any]:
    if source_name == "stock_info_global_cls":
        return {"source_name": source_name, "symbol": "全部"}
    return {"source_name": source_name}


def static_source_kwargs(source_name: str) -> dict[str, Any]:
    default_kwargs: dict[str, dict[str, Any]] = {
        "stock_info_a_code_name": {},
        "stock_info_sh_name_code": {"symbol": "主板A股"},
        "stock_info_sz_name_code": {"symbol": "A股列表"},
        "stock_info_sz_change_name": {"symbol": "全称变更"},
        "stock_info_sh_delist": {"symbol": "全部"},
        "stock_info_sz_delist": {"symbol": "终止上市公司"},
    }
    return {"source_name": source_name, **default_kwargs.get(source_name, {})}


def maybe_sleep(*, index: int, total: int, sleep_seconds: float) -> None:
    if sleep_seconds > 0 and index + 1 < total:
        time.sleep(sleep_seconds)


def window_label(trade_date: str) -> str:
    if PREWARM_WINDOW.start <= trade_date <= PREWARM_WINDOW.end:
        return "prewarm_20"
    if EVAL_WINDOW.start <= trade_date <= EVAL_WINDOW.end:
        return "eval_240"
    return "outside_fixed_windows"


if __name__ == "__main__":
    main()
