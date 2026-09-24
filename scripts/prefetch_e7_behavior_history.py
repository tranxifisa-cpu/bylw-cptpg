"""Resume-safe E7 history cache.

The existing daily_basic, fina_indicator and index_weight caches contain the
historical factors. --prices-only adds missing daily OHLCV and adj_factor rows
by trade date; --verify-prices checks every open day without contacting Tushare.
Adjusted returns are constructed downstream from raw closes and adj_factor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvp_cpt_pg.raw_cache import DataFrameCache


OPPORTUNITIES = ROOT / "results/e7_heterogeneous_agents_offer_value/e6_opportunities.parquet"
CACHE = ROOT / "artifacts/cache/raw/tushare"
BASIC_FIELDS = (
    "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,"
    "dv_ratio,total_mv,circ_mv"
)
FINANCIAL_FIELDS = "ts_code,ann_date,end_date,roe,roa,roic"
INDEX_FIELDS = "index_code,con_code,trade_date,weight"
ADJ_FIELDS = "ts_code,trade_date,adj_factor"


def parse_date(value: str) -> date:
    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError(f"Expected YYYYMMDD, got {value!r}")
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def stamp(value: date) -> str:
    return value.strftime("%Y%m%d")


def month_windows(start: date, end: date):
    cursor = start.replace(day=1)
    while cursor <= end:
        following = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield max(cursor, start), min(following - timedelta(days=1), end)
        cursor = following


def uncovered(start: date, end: date, covered: list[tuple[date, date]]):
    cursor = start
    for left, right in sorted(covered):
        if right < cursor or left > end:
            continue
        if left > cursor:
            yield cursor, min(left - timedelta(days=1), end)
        cursor = max(cursor, right + timedelta(days=1))
        if cursor > end:
            return
    if cursor <= end:
        yield cursor, end


def cached(cache: DataFrameCache, namespace: str, payload: dict) -> bool:
    record = cache._record(namespace, payload)
    return record.data_path.exists() and record.meta_path.exists()


def cached_intervals(cache: DataFrameCache, namespace: str, *, key: str,
                     fields: str) -> dict[str, list[tuple[date, date]]]:
    intervals: dict[str, list[tuple[date, date]]] = {}
    for path in (cache.root / namespace).glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            code = payload.get(key)
            if (not code or payload.get("fields") != fields
                    or not path.with_suffix(".pkl").exists()):
                continue
            intervals.setdefault(code, []).append(
                (parse_date(payload["start_date"]), parse_date(payload["end_date"])))
        except (OSError, ValueError, KeyError, json.JSONDecodeError, argparse.ArgumentTypeError):
            continue
    return intervals


def cached_index_months(cache: DataFrameCache) -> set[str]:
    months = set()
    for path in (cache.root / "index_weight").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            data_path = path.with_suffix(".pkl")
            if (payload.get("index_code") != "000300.SH"
                    or payload.get("fields") != INDEX_FIELDS or not data_path.exists()):
                continue
            frame = pd.read_pickle(data_path)
            if frame.empty or not {"trade_date", "con_code"}.issubset(frame.columns):
                continue
            frame = frame.copy()
            frame["month"] = frame.trade_date.astype(str).str[:6]
            for month, group in frame.groupby("month"):
                if group.groupby("trade_date").con_code.nunique().ge(300).any():
                    months.add(month)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return months


def behavior_codes(path: Path) -> list[str]:
    frame = pd.read_parquet(path, columns=["stock_symbol"])
    symbols = frame.stock_symbol.dropna().astype(str).unique()
    invalid = [symbol for symbol in symbols if not re.fullmatch(r"(?:SH|SZ)\d{6}", symbol)]
    if invalid:
        raise ValueError(f"Unsupported behavior stock code, example: {invalid[0]}")
    return sorted(f"{symbol[2:]}.{symbol[:2]}" for symbol in symbols)


def price_payloads(day: str) -> tuple[dict, dict]:
    return dict(trade_date=day), dict(trade_date=day, fields=ADJ_FIELDS)


def price_gaps(cache: DataFrameCache, days: list[str]) -> tuple[list[str], list[str]]:
    missing_daily, missing_adjustment = [], []
    for day in days:
        daily, adjustment = price_payloads(day)
        if not cached(cache, "daily", daily):
            missing_daily.append(day)
        if not cached(cache, "adj_factor", adjustment):
            missing_adjustment.append(day)
    return missing_daily, missing_adjustment


def open_days(calendar: pd.DataFrame) -> list[str]:
    return sorted(calendar.loc[calendar.is_open.astype(str).eq("1"), "cal_date"].astype(str))


def tushare_token() -> str | None:
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("Tushare_Token")
    if token or os.name != "nt":
        return token
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
        for name in ("TUSHARE_TOKEN", "Tushare_Token"):
            try:
                token, _ = winreg.QueryValueEx(key, name)
            except FileNotFoundError:
                continue
            if token:
                return str(token)
    return None


def check_price_cache(cache: DataFrameCache, days: list[str]) -> None:
    missing_daily, missing_adjustment = price_gaps(cache, days)
    if missing_daily or missing_adjustment:
        raise RuntimeError(
            f"Price cache incomplete: daily={len(missing_daily)} "
            f"adj_factor={len(missing_adjustment)}; "
            f"first missing daily={missing_daily[:3]} adjustment={missing_adjustment[:3]}"
        )
    for day in days:
        daily, adjustment = price_payloads(day)
        for namespace, payload, fields in (
            ("daily", daily, {"ts_code", "trade_date", "open", "close", "pre_close", "vol"}),
            ("adj_factor", adjustment, {"ts_code", "trade_date", "adj_factor"}),
        ):
            frame = pd.read_pickle(cache._record(namespace, payload).data_path)
            if frame.empty or not fields.issubset(frame.columns):
                raise RuntimeError(f"Invalid {namespace} cache on {day}")
            if not frame.trade_date.astype(str).eq(day).all() or frame.ts_code.duplicated().any():
                raise RuntimeError(f"Mismatched or duplicate {namespace} rows on {day}")
    print(f"price_cache_complete open_days={len(days)} daily={len(days)} "
          f"adj_factor={len(days)}", flush=True)


class Client:
    def __init__(self, token: str, cache: DataFrameCache, per_minute: int, attempts: int):
        self.token = token
        self.api = ts.pro_api(token)
        self.cache = cache
        self.interval = 60.0 / per_minute
        self.next_call = 0.0
        self.attempts = attempts
        self.requests = 0

    def fetch(self, namespace: str, payload: dict, api_method: str, *, required: set[str],
              nonempty: bool = True, row_limit: int | None = None) -> tuple[pd.DataFrame, bool]:
        record = self.cache._record(namespace, payload)
        if record.data_path.exists() and not record.meta_path.exists():
            # A prior interruption after saving data must not trigger a second API call.
            pd.read_pickle(record.data_path)
            record.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if cached(self.cache, namespace, payload):
            return pd.read_pickle(record.data_path), False
        for attempt in range(self.attempts):
            delay = self.next_call - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_call = time.monotonic() + self.interval
            self.requests += 1
            try:
                frame = getattr(self.api, api_method)(**payload)
                if not isinstance(frame, pd.DataFrame):
                    raise ValueError("Tushare did not return a DataFrame")
                if nonempty and frame.empty:
                    raise ValueError("Tushare returned no rows for an expected open period")
                if not frame.empty and not required.issubset(frame.columns):
                    raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
                if row_limit is not None and len(frame) >= row_limit:
                    raise TruncatedResponse
                if namespace == "daily_basic" and len(frame) >= 6000:
                    raise ValueError("Daily-basic response reached its 6000-row cap")
                if namespace == "index_weight":
                    counts = frame.groupby("trade_date").con_code.nunique()
                    if not counts.ge(300).any():
                        raise ValueError("No complete 300-constituent snapshot returned")
                temporary = record.data_path.with_suffix(".pkl.part")
                frame.to_pickle(temporary)
                temporary.replace(record.data_path)
                record.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return frame, True
            except TruncatedResponse:
                raise
            except Exception as exc:
                if attempt + 1 == self.attempts:
                    detail = str(exc).replace(self.token, "[redacted]")
                    raise RuntimeError(
                        f"{namespace} failed after {self.attempts} attempts; "
                        f"rerun the same command to resume at this item: {detail[:200]}"
                    ) from None
                time.sleep(min(60.0, 2.0 ** attempt * 5.0))
        raise AssertionError("unreachable")


class TruncatedResponse(Exception):
    pass


def fetch_financial(client: Client, code: str, left: date, right: date) -> tuple[int, int]:
    payload = dict(ts_code=code, start_date=stamp(left), end_date=stamp(right),
                   fields=FINANCIAL_FIELDS)
    try:
        frame, fetched = client.fetch("fina_indicator", payload, "fina_indicator",
                                      required={"ts_code", "ann_date", "roe", "roa", "roic"},
                                      nonempty=False, row_limit=100)
        return int(fetched), int(frame.empty)
    except TruncatedResponse:
        if left >= right:
            raise RuntimeError(f"Financial records exceed 100 rows on {code} {stamp(left)}")
        middle = left + (right - left) // 2
        first = fetch_financial(client, code, left, middle)
        second = fetch_financial(client, code, middle + timedelta(days=1), right)
        return first[0] + second[0], first[1] + second[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opportunities", type=Path, default=OPPORTUNITIES)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--start-date", type=parse_date, default=parse_date("20160501"),
                        help="Daily-basic start, including lookback before first behavior event")
    parser.add_argument("--end-date", type=parse_date, default=parse_date("20230831"))
    parser.add_argument("--fundamental-start", type=parse_date,
                        default=parse_date("20150101"))
    parser.add_argument("--index-start", type=parse_date, default=parse_date("20160101"))
    parser.add_argument("--rate-per-minute", type=int, default=60)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--plan", action="store_true", help="Print scope without querying Tushare")
    parser.add_argument("--prices-only", action="store_true",
                        help="Fetch only missing daily OHLCV and adjustment factors")
    parser.add_argument("--verify-prices", action="store_true",
                        help="Check cached price dates and row schemas without using Tushare")
    args = parser.parse_args()
    if args.start_date > args.end_date or args.fundamental_start > args.end_date:
        parser.error("Start dates must not exceed end date")
    if args.index_start > args.end_date or args.rate_per_minute < 1 or args.attempts < 1:
        parser.error("Invalid index start, rate or retry count")
    if not args.opportunities.is_file():
        parser.error(f"Missing behavior opportunities: {args.opportunities}")

    codes = behavior_codes(args.opportunities)
    cache = DataFrameCache(args.cache)
    calendar_payload = dict(start_date=stamp(args.start_date), end_date=stamp(args.end_date))
    calendar_record = cache._record("trade_cal", calendar_payload)
    if args.plan or args.verify_prices:
        if not cached(cache, "trade_cal", calendar_payload):
            parser.error("Cached trade calendar is required for --plan/--verify-prices")
        days = open_days(pd.read_pickle(calendar_record.data_path))
        missing_daily, missing_adjustment = price_gaps(cache, days)
        print(f"price_window={stamp(args.start_date)}..{stamp(args.end_date)} "
              f"open_days={len(days)} missing_daily={len(missing_daily)} "
              f"missing_adj_factor={len(missing_adjustment)}", flush=True)
        if args.verify_prices:
            check_price_cache(cache, days)
            return
        if args.prices_only:
            return
    if args.prices_only:
        financial_tasks = []
        index_tasks = []
    else:
        financial_cached = cached_intervals(cache, "fina_indicator", key="ts_code",
                                            fields=FINANCIAL_FIELDS)
        financial_tasks = []
        for code in codes:
            prior = financial_cached.get(code, [])
            financial_tasks.extend((code, left, right) for left, right in
                                   uncovered(args.fundamental_start, args.end_date, prior))
        index_cached = cached_index_months(cache)
        index_tasks = []
        for left, right in month_windows(args.index_start, args.end_date):
            payload = dict(index_code="000300.SH", start_date=stamp(left),
                           end_date=stamp(right), fields=INDEX_FIELDS)
            if stamp(left)[:6] not in index_cached:
                index_tasks.append(payload)

    print(f"mode={'prices-only' if args.prices_only else 'all'} "
          f"behavior_stocks={len(codes)} financial_queries_pending={len(financial_tasks)} "
          f"index_months_pending={len(index_tasks)} "
          f"daily_basic_window={stamp(args.start_date)}..{stamp(args.end_date)}", flush=True)
    if args.plan:
        return

    token = tushare_token()
    if not token:
        parser.error("TUSHARE_TOKEN is absent from this process and the Windows Machine scope")
    client = Client(token, cache, args.rate_per_minute, args.attempts)
    calendar, _ = client.fetch("trade_cal", calendar_payload,
                               "trade_cal", required={"cal_date", "is_open"})
    days = open_days(calendar)
    fetched_daily = 0
    fetched_adjustment = 0
    for i, day in enumerate(days, 1):
        daily, adjustment = price_payloads(day)
        _, fetched = client.fetch("daily", daily, "daily",
                                  required={"ts_code", "trade_date", "open", "close", "pre_close", "vol"},
                                  row_limit=6000)
        fetched_daily += int(fetched)
        _, fetched = client.fetch("adj_factor", adjustment, "adj_factor",
                                  required={"ts_code", "trade_date", "adj_factor"},
                                  row_limit=6000)
        fetched_adjustment += int(fetched)
        if i % 50 == 0 or i == len(days):
            print(f"prices {i}/{len(days)} new_daily={fetched_daily} "
                  f"new_adj_factor={fetched_adjustment}", flush=True)

    fetched_basic = 0
    for i, day in enumerate([] if args.prices_only else days, 1):
        payload = dict(trade_date=day, fields=BASIC_FIELDS)
        _, fetched = client.fetch("daily_basic", payload, "daily_basic",
                                  required={"ts_code", "trade_date", "pb", "turnover_rate",
                                            "volume_ratio", "circ_mv"})
        fetched_basic += int(fetched)
        if i % 50 == 0 or i == len(days):
            print(f"daily_basic {i}/{len(days)} newly_fetched={fetched_basic}", flush=True)

    fetched_financial = 0
    empty_financial = 0
    for i, (code, left, right) in enumerate(financial_tasks, 1):
        fetched, empty = fetch_financial(client, code, left, right)
        fetched_financial += fetched
        empty_financial += empty
        if i % 100 == 0 or i == len(financial_tasks):
            print(f"fina_indicator {i}/{len(financial_tasks)} "
                  f"newly_fetched={fetched_financial} empty={empty_financial}", flush=True)

    fetched_index = 0
    for i, payload in enumerate(index_tasks, 1):
        _, fetched = client.fetch("index_weight", payload, "index_weight",
                                  required={"index_code", "con_code", "trade_date", "weight"})
        fetched_index += int(fetched)
        if i % 12 == 0 or i == len(index_tasks):
            print(f"index_weight {i}/{len(index_tasks)} "
                  f"newly_fetched={fetched_index}", flush=True)

    check_price_cache(cache, days)
    print(f"complete: stocks={len(codes)} open_dates={len(days)} "
          f"new_daily={fetched_daily} new_adj_factor={fetched_adjustment} "
          f"new_daily_basic={fetched_basic} new_fina_indicator={fetched_financial} "
          f"empty_fina_indicator={empty_financial} new_index_months={fetched_index} "
          f"api_calls_including_retries={client.requests}", flush=True)


if __name__ == "__main__":
    main()
