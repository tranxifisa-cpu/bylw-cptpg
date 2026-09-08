from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .progress import progress
from .raw_cache import DataFrameCache
from .utils import get_env_var, zscore


def _load_tushare() -> Any:
    return import_module("tushare")


@dataclass
class MarketDataset:
    universe: pd.DataFrame
    panel: pd.DataFrame
    source_status: pd.DataFrame
    trade_dates: list[str]

    def stock_state(self, trade_date: str) -> pd.DataFrame:
        return (
            self.panel[self.panel["trade_date"] == trade_date]
            .sort_values("ts_code")
            .reset_index(drop=True)
        )

    def previous_trade_date(self, trade_date: str) -> str:
        previous_dates = [date for date in self.trade_dates if date < trade_date]
        if not previous_dates:
            raise RuntimeError(f"No observable market state is available before {trade_date}")
        return previous_dates[-1]

    def observed_stock_state(self, trade_date: str) -> pd.DataFrame:
        return self.stock_state(self.previous_trade_date(trade_date))


class TushareDataClient:
    def __init__(self, config: ExperimentConfig) -> None:
        token = get_env_var("Tushare_Token", "TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("Tushare token is not configured")
        self.config = config
        ts = _load_tushare()
        self.pro = ts.pro_api(token)
        self.cache = DataFrameCache(config.raw_cache_dir / "tushare")

    def _cached(self, namespace: str, payload: dict[str, Any], fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        return self.cache.get_or_fetch(namespace, payload, fn)

    def stock_basic(self) -> pd.DataFrame:
        return self._cached(
            "stock_basic",
            {"fields": "ts_code,symbol,name,area,industry,market,list_date"},
            lambda: self.pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,market,list_date",
            ),
        )

    def stock_basic_by_status(self, list_status: str) -> pd.DataFrame:
        fields = "ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status"
        return self._cached(
            "stock_basic",
            {"list_status": list_status, "fields": fields},
            lambda: self.pro.stock_basic(
                exchange="",
                list_status=list_status,
                fields=fields,
            ),
        )

    def stock_basic_all_status(self) -> pd.DataFrame:
        frames = []
        for list_status in ("L", "D", "P"):
            frame = self.stock_basic_by_status(list_status).copy()
            if "list_status" not in frame.columns:
                frame["list_status"] = list_status
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        output = pd.concat(frames, ignore_index=True)
        return output.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)

    def trade_calendar(self, start_date: str, end_date: str) -> list[str]:
        df = self._cached(
            "trade_cal",
            {"start_date": start_date, "end_date": end_date},
            lambda: self.pro.trade_cal(exchange="", start_date=start_date, end_date=end_date),
        )
        df = df[df["is_open"].astype(str) == "1"].copy()
        return sorted(df["cal_date"].astype(str).tolist())

    def daily_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        return self._cached(
            "daily",
            {"trade_date": trade_date},
            lambda: self.pro.daily(trade_date=trade_date),
        )

    def daily_basic_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        df = self._cached(
            "daily_basic",
            {
                "trade_date": trade_date,
                "fields": "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,dv_ratio,total_mv,circ_mv",
            },
            lambda: self.pro.daily_basic(
                trade_date=trade_date,
                fields="ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,dv_ratio,total_mv,circ_mv",
            ),
        )
        expected_columns = [
            "ts_code",
            "trade_date",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pb",
            "ps",
            "dv_ratio",
            "total_mv",
            "circ_mv",
        ]
        if df.empty and "ts_code" not in df.columns:
            return pd.DataFrame(columns=expected_columns)
        return df.reindex(columns=expected_columns)

    def index_weight(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        fields = "index_code,con_code,trade_date,weight"
        return self._cached(
            "index_weight",
            {
                "index_code": index_code,
                "start_date": start_date,
                "end_date": end_date,
                "fields": fields,
            },
            lambda: self.pro.index_weight(
                index_code=index_code,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
            ),
        ).reindex(columns=fields.split(","))

class MarketDatasetBuilder:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.tushare = TushareDataClient(config)

    def build(self) -> MarketDataset:
        lookback_start = (pd.Timestamp(self.config.prewarm.start) - timedelta(days=120)).strftime("%Y%m%d")
        trade_dates = self.tushare.trade_calendar(lookback_start, self.config.evaluation.end)
        print(f"Loaded {len(trade_dates)} trade dates")
        universe_by_date = self._load_universe_by_date()
        universe = self._build_universe(universe_by_date)
        universe_codes = universe["ts_code"].tolist()
        print(f"Selected universe size: {len(universe_codes)}")
        panel = self._build_stock_panel(universe_codes, trade_dates, universe_by_date)
        if self.config.strict_drop_missing_stocks:
            trade_dates = [date for date in trade_dates if date in set(panel["trade_date"].astype(str))]
            universe_codes = sorted(panel["ts_code"].astype(str).unique().tolist())
            universe = universe[universe["ts_code"].isin(universe_codes)].sort_values("ts_code").reset_index(drop=True)
            print(f"Strict clean universe size: {len(universe_codes)}")
        source_status = pd.DataFrame()
        if self.config.strict_drop_missing_stocks:
            clean_status = pd.DataFrame([
                {
                    "source": "strict_clean_missing_stock_drop",
                    "success": 1,
                    "rows": int(panel.attrs.get("strict_clean_kept_stocks", len(universe_codes))),
                    "error": (
                        f"removed_stocks={panel.attrs.get('strict_clean_removed_stocks', 0)};"
                        f"strict_dates={panel.attrs.get('strict_clean_dates', 0)}"
                    ),
                }
            ])
            source_status = pd.concat([source_status, clean_status], ignore_index=True)
        panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        return MarketDataset(
            universe=universe,
            panel=panel,
            source_status=source_status,
            trade_dates=trade_dates,
        )

    def _load_universe_by_date(self) -> pd.DataFrame | None:
        if self.config.universe_by_date_path is None:
            return None
        path = self.config.universe_by_date_path
        if not path.exists():
            raise FileNotFoundError(f"Universe-by-date cache is missing: {path}")
        frame = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
        required = {"trade_date", "ts_code"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Universe-by-date cache is missing required columns: {sorted(missing)}")
        frame = frame.dropna(subset=["trade_date", "ts_code"]).copy()
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["ts_code"] = frame["ts_code"].astype(str)
        return frame.drop_duplicates(subset=["trade_date", "ts_code"]).reset_index(drop=True)

    def _build_universe(self, universe_by_date: pd.DataFrame | None = None) -> pd.DataFrame:
        stock_basic = self.tushare.stock_basic_all_status().copy() if universe_by_date is not None else self.tushare.stock_basic().copy()
        stock_basic["industry"] = stock_basic["industry"].fillna("未知行业")
        stock_basic["list_date"] = stock_basic["list_date"].astype(str)
        if universe_by_date is None and self.config.index_universe_code is not None:
            return self._build_index_universe(stock_basic)
        listing_cutoff = self.config.evaluation.end if universe_by_date is not None else self.config.prewarm.end
        stock_basic = stock_basic[stock_basic["list_date"] <= listing_cutoff]
        stock_basic = stock_basic[~stock_basic["industry"].isin(self.config.finance_industries)].copy()
        if universe_by_date is not None:
            allowed_codes = set(universe_by_date["ts_code"].astype(str))
            stock_basic = stock_basic[stock_basic["ts_code"].astype(str).isin(allowed_codes)].copy()
            if stock_basic.empty:
                raise RuntimeError("Universe-by-date cache produced an empty stock universe")
            return stock_basic.sort_values("ts_code").reset_index(drop=True)
        if self.config.universe_size is None and self.config.universe_industry_cap is None:
            return stock_basic.sort_values("ts_code").reset_index(drop=True)
        recent_dates = self.tushare.trade_calendar(self.config.prewarm.start, self.config.prewarm.end)[-20:]
        daily_frames = []
        valid_codes = set(stock_basic["ts_code"])
        for trade_date in progress(recent_dates, desc="universe liquidity", total=len(recent_dates)):
            df = self.tushare.daily_by_trade_date(trade_date)
            if df.empty:
                continue
            df = df[df["ts_code"].isin(valid_codes)].copy()
            daily_frames.append(df[["ts_code", "amount"]])
        if not daily_frames:
            raise RuntimeError("Unable to build universe because no recent daily data was loaded")
        liquidity = pd.concat(daily_frames, ignore_index=True)
        liquidity["amount"] = pd.to_numeric(liquidity["amount"], errors="coerce").fillna(0.0)
        liquidity = liquidity.groupby("ts_code", as_index=False)["amount"].mean().rename(columns={"amount": "avg_amount_20d"})
        ranked = stock_basic.merge(liquidity, on="ts_code", how="inner").sort_values("avg_amount_20d", ascending=False)
        chosen_rows: list[pd.Series] = []
        industry_counts: dict[str, int] = {}
        target_size = self.config.universe_size or len(ranked)
        for _, row in ranked.iterrows():
            industry = str(row["industry"])
            if self.config.universe_industry_cap is not None and industry_counts.get(industry, 0) >= self.config.universe_industry_cap:
                continue
            chosen_rows.append(row)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            if len(chosen_rows) >= target_size:
                break
        universe = pd.DataFrame(chosen_rows).reset_index(drop=True)
        if len(universe) < target_size:
            raise RuntimeError(f"Universe selection returned only {len(universe)} stocks")
        return universe

    def _build_index_universe(self, stock_basic: pd.DataFrame) -> pd.DataFrame:
        weights = self.tushare.index_weight(
            self.config.index_universe_code,
            self.config.prewarm.start,
            self.config.prewarm.end,
        )
        if weights.empty:
            raise RuntimeError(
                f"Tushare index_weight returned no constituents for "
                f"{self.config.index_universe_code} between {self.config.prewarm.start} and {self.config.prewarm.end}"
            )
        weights = weights.dropna(subset=["con_code", "trade_date"]).copy()
        weights["trade_date"] = weights["trade_date"].astype(str)
        latest_date = str(weights["trade_date"].max())
        latest_weights = weights[weights["trade_date"] == latest_date].copy()
        latest_codes = latest_weights["con_code"].astype(str).drop_duplicates().tolist()
        if not latest_codes:
            raise RuntimeError(f"Index universe {self.config.index_universe_code} has no constituent codes on {latest_date}")
        stock_basic = stock_basic[stock_basic["ts_code"].astype(str).isin(latest_codes)].copy()
        if stock_basic.empty:
            raise RuntimeError(f"Index universe {self.config.index_universe_code} is not covered by stock_basic")
        stock_basic["index_universe_code"] = self.config.index_universe_code
        stock_basic["index_universe_date"] = latest_date
        return stock_basic.sort_values("ts_code").reset_index(drop=True)

    def _build_stock_panel(
        self,
        universe_codes: list[str],
        trade_dates: list[str],
        universe_by_date: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        daily_frames: list[pd.DataFrame] = []
        basic_frames: list[pd.DataFrame] = []
        skipped_basic_dates: list[str] = []
        universe_set = set(universe_codes)
        universe_by_date_map: dict[str, set[str]] = {}
        if universe_by_date is not None:
            universe_by_date_map = {
                trade_date: set(frame["ts_code"].astype(str))
                for trade_date, frame in universe_by_date.groupby("trade_date")
            }
        for trade_date in progress(trade_dates, desc="daily market", total=len(trade_dates)):
            daily = self.tushare.daily_by_trade_date(trade_date)
            allowed_codes = universe_by_date_map.get(trade_date, universe_set)
            daily_frames.append(daily[daily["ts_code"].astype(str).isin(allowed_codes)].copy())
            daily_basic = self.tushare.daily_basic_by_trade_date(trade_date)
            if self.config.strict_drop_missing_stocks and daily_basic.empty and self.config.prewarm.start <= trade_date <= self.config.evaluation.end:
                skipped_basic_dates.append(trade_date)
                continue
            basic_frames.append(daily_basic[daily_basic["ts_code"].astype(str).isin(allowed_codes)].copy())
        daily = pd.concat(daily_frames, ignore_index=True)
        daily_basic = pd.concat(basic_frames, ignore_index=True)
        if skipped_basic_dates:
            daily = daily[~daily["trade_date"].astype(str).isin(skipped_basic_dates)].copy()
            print(f"Strict clean skipped empty daily_basic dates: {len(skipped_basic_dates)}")
        panel = daily.merge(daily_basic, on=["ts_code", "trade_date"], how="left", suffixes=("", "_basic"))
        panel["trade_date_dt"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d")
        panel = panel.sort_values(["ts_code", "trade_date_dt"]).reset_index(drop=True)
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pb",
            "ps",
            "dv_ratio",
            "total_mv",
            "circ_mv",
        ]
        for col in numeric_cols:
            if col in panel.columns:
                panel[col] = pd.to_numeric(panel[col], errors="coerce", downcast="float")
        panel["ret_1d"] = panel["pct_chg"] / 100.0
        panel["open_close_ret"] = panel["close"] / panel["open"] - 1.0
        panel["ret_5d"] = panel.groupby("ts_code")["close"].transform(lambda s: s / s.shift(5) - 1.0)
        panel["vol_20d"] = panel.groupby("ts_code")["ret_1d"].transform(lambda s: s.rolling(20, min_periods=5).std())
        panel["amount_20d_avg"] = panel.groupby("ts_code")["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        panel = _add_qbot_technical_factors(panel)
        if self.config.strict_drop_missing_stocks:
            panel = self._drop_stocks_with_missing_required_fields(panel)
        panel["ret_5d"] = panel["ret_5d"].fillna(0.0)
        panel["vol_20d"] = panel["vol_20d"].fillna(panel.groupby("ts_code")["ret_1d"].transform("std")).fillna(0.0)
        panel["amount_20d_avg"] = panel["amount_20d_avg"].fillna(panel["amount"])
        for col in ("pe", "pb", "ps", "dv_ratio", "turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"):
            if col not in panel.columns:
                panel[col] = 0.0
            panel[col] = pd.to_numeric(panel[col], errors="coerce", downcast="float").fillna(0.0)
        for col in ("qbot_boll_z", "qbot_rsi_14", "qbot_macd_hist", "qbot_rsrs_beta"):
            panel[col] = pd.to_numeric(panel[col], errors="coerce", downcast="float").fillna(0.0)
        panel = _downcast_panel_float_columns(panel)
        panel = panel.drop(columns=["trade_date_dt"])
        return panel

    def _drop_stocks_with_missing_required_fields(self, panel: pd.DataFrame) -> pd.DataFrame:
        strict_window = panel[
            (panel["trade_date"].astype(str) >= self.config.prewarm.start)
            & (panel["trade_date"].astype(str) <= self.config.evaluation.end)
        ].copy()
        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "pct_chg",
            "amount",
            "ret_1d",
            "open_close_ret",
            "ret_5d",
            "vol_20d",
            "amount_20d_avg",
            "pe",
            "pb",
            "ps",
            "dv_ratio",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "total_mv",
            "circ_mv",
            "qbot_boll_z",
            "qbot_rsi_14",
            "qbot_macd_hist",
            "qbot_rsrs_beta",
        ]
        missing_columns = [col for col in required_columns if col not in strict_window.columns]
        if missing_columns:
            raise RuntimeError(f"Strict clean mode is missing required columns: {missing_columns}")
        expected_dates = strict_window["trade_date"].nunique()
        row_counts = strict_window.groupby("ts_code")["trade_date"].nunique()
        complete_date_codes = set(row_counts[row_counts == expected_dates].index)
        missing_by_code = strict_window.groupby("ts_code")[required_columns].apply(lambda frame: bool(frame.isna().any().any()))
        clean_codes = sorted(
            code
            for code in complete_date_codes
            if code in missing_by_code.index and not bool(missing_by_code.loc[code])
        )
        if not clean_codes:
            raise RuntimeError("Strict clean mode removed every stock; check the required field list or date range")
        removed_count = panel["ts_code"].nunique() - len(clean_codes)
        print(f"Strict clean removed stocks with missing required fields: {removed_count}")
        cleaned = panel[panel["ts_code"].isin(clean_codes)].copy()
        cleaned.attrs["strict_clean_removed_stocks"] = int(removed_count)
        cleaned.attrs["strict_clean_kept_stocks"] = int(len(clean_codes))
        cleaned.attrs["strict_clean_dates"] = int(expected_dates)
        return cleaned

def factorize_state(state: pd.DataFrame) -> pd.DataFrame:
    frame = state.copy()
    frame["momentum_score"] = zscore(frame["ret_5d"]) + 0.5 * zscore(frame["ret_1d"])
    value_pe = -zscore(frame["pe"].replace(0, np.nan).fillna(frame["pe"].median()))
    value_pb = -zscore(frame["pb"].replace(0, np.nan).fillna(frame["pb"].median()))
    value_ps = -zscore(frame["ps"].replace(0, np.nan).fillna(frame["ps"].median()))
    dividend_score = zscore(frame["dv_ratio"].replace(0, np.nan).fillna(0.0))
    liquidity_score = zscore(frame["amount_20d_avg"]) + 0.5 * zscore(frame["circ_mv"].replace(0, np.nan).fillna(frame["circ_mv"].median()))
    stability_score = -zscore(frame["vol_20d"]) - 0.25 * zscore(frame["volume_ratio"].replace(0, np.nan).fillna(1.0))
    frame["value_score"] = value_pe + value_pb + 0.5 * value_ps + 0.25 * dividend_score
    frame["quality_score"] = 0.6 * liquidity_score + 0.8 * stability_score + 0.2 * dividend_score
    frame["low_vol_score"] = -zscore(frame["vol_20d"]) + 0.1 * zscore(frame["amount_20d_avg"])
    frame["mean_reversion_score"] = -zscore(frame["ret_5d"]) - 0.5 * zscore(frame["ret_1d"])
    frame["qbot_boll_reversion_score"] = zscore(-frame["qbot_boll_z"]) + 0.25 * zscore(-frame["ret_5d"])
    frame["qbot_rsi_reversal_score"] = zscore(50.0 - frame["qbot_rsi_14"]) + 0.25 * zscore(-frame["ret_5d"])
    frame["qbot_macd_trend_score"] = zscore(frame["qbot_macd_hist"]) + 0.25 * zscore(frame["ret_5d"])
    frame["qbot_rsrs_timing_score"] = zscore(frame["qbot_rsrs_beta"]) + 0.25 * zscore(frame["ret_5d"])
    frame["balanced_score"] = (
        frame["momentum_score"]
        + frame["value_score"]
        + frame["quality_score"]
        + frame["low_vol_score"]
        + frame["qbot_macd_trend_score"]
        + frame["qbot_rsrs_timing_score"]
    ) / 6.0
    return frame


def _add_qbot_technical_factors(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    grouped = frame.groupby("ts_code", group_keys=False)
    close = grouped["close"]
    high = grouped["high"]
    low = grouped["low"]
    frame["qbot_ma_20"] = close.transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["qbot_close_std_20"] = close.transform(lambda s: s.rolling(20, min_periods=5).std())
    denom = frame["qbot_close_std_20"].replace(0, np.nan)
    frame["qbot_boll_z"] = (frame["close"] - frame["qbot_ma_20"]) / denom
    frame["qbot_rsi_14"] = close.transform(_rsi_14)
    ema_12 = close.transform(lambda s: s.ewm(span=12, adjust=False, min_periods=3).mean())
    ema_26 = close.transform(lambda s: s.ewm(span=26, adjust=False, min_periods=5).mean())
    frame["qbot_macd_diff"] = ema_12 - ema_26
    frame["qbot_macd_dea"] = frame.groupby("ts_code")["qbot_macd_diff"].transform(
        lambda s: s.ewm(span=9, adjust=False, min_periods=3).mean()
    )
    frame["qbot_macd_hist"] = frame["qbot_macd_diff"] - frame["qbot_macd_dea"]
    frame["qbot_rsrs_beta"] = frame.groupby("ts_code", group_keys=False)[["high", "low"]].apply(_rsrs_beta).reset_index(level=0, drop=True)
    return frame


def _downcast_panel_float_columns(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    float_columns = frame.select_dtypes(include=["float64"]).columns
    for column in float_columns:
        frame[column] = frame[column].astype("float32", copy=False)
    return frame


def _rsi_14(close: pd.Series) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0.0).rolling(14, min_periods=5).mean()
    loss = (-diff.clip(upper=0.0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def _rsrs_beta(frame: pd.DataFrame) -> pd.Series:
    cov = frame["high"].rolling(18, min_periods=6).cov(frame["low"])
    var = frame["low"].rolling(18, min_periods=6).var()
    return (cov / var.replace(0, np.nan)).fillna(0.0)
