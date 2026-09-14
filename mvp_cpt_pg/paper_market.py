"""Offline, point-in-time inputs for the seven-block paper experiment plan."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .synthetic_env import _regime_labels, _semi_synthetic_regime_parameters


FEATURES = ["bias", "cash", "momentum_signal", "reversal_signal",
            "low_vol_signal", "value_signal", "turnover_signal",
            "quality_signal", "cycle", "previous_target"]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cache_records(root, namespace):
    for path in sorted((Path(root) / namespace).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = path.with_suffix(".pkl")
        if data.exists():
            yield payload, data


def dated_cache(root, namespace, required_columns=()):
    required = set(required_columns)
    candidates = {}
    for payload, path in cache_records(root, namespace):
        if "trade_date" in payload:
            date = str(payload["trade_date"])
            candidates.setdefault(date, [])
            fields = payload.get("fields")
            if fields and not required.issubset({field.strip() for field in fields.split(",")}):
                continue
            candidates[date].append(path)
    result = {}
    for date, paths in candidates.items():
        if not paths:
            raise ValueError(f"No {namespace} cache with required columns {sorted(required)} for {date}")
        if len(paths) == 1:
            frame = pd.read_pickle(paths[0])
            if frame.empty:
                raise ValueError(f"Empty {namespace} cache for {date}: {paths[0].name}; refresh this cache")
            if not required.issubset(frame.columns):
                raise ValueError(f"Incomplete {namespace} cache for {date}: {paths[0].name}")
            result[date] = paths[0]
            continue
        # Request-field variants are normal. Compare usable duplicates by asset/date,
        # rather than choosing the newest file or silently merging conflicting values.
        frames = []
        for path in paths:
            frame = pd.read_pickle(path)
            if frame.empty:
                continue
            if not required.issubset(frame.columns):
                continue
            keys = ["ts_code", "trade_date"]
            if not set(keys).issubset(frame.columns) or frame.duplicated(keys).any():
                raise ValueError(f"Invalid {namespace} cache keys for {date}: {path}")
            if set(frame.trade_date.astype(str)) != {date}:
                raise ValueError(f"{namespace} cache contents disagree with date {date}: {path}")
            frames.append((path, frame.set_index(keys).sort_index()))
        if not frames:
            raise ValueError(f"No usable {namespace} cache for {date}: required {sorted(required)}")
        reference_path, reference = frames[0]
        for path, frame in frames[1:]:
            columns = sorted(required - {"ts_code", "trade_date"}) if required else sorted(reference.columns)
            try:
                pd.testing.assert_frame_equal(reference[columns], frame[columns],
                                              check_dtype=False, check_exact=True)
            except (AssertionError, KeyError) as error:
                raise ValueError(f"Conflicting {namespace} cache data for {date}: "
                                 f"{reference_path.name} vs {path.name}") from error
        result[date] = max(frames, key=lambda item: len(item[1].columns))[0]
    return result


def prepare_panel(root, output, start, end, assets=30, pool_seed=2022,
                  allow_suspension_carry=False):
    """Select before start; never replace stocks using future completeness."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    sources, weight_frames = [], []
    for payload, path in cache_records(root, "index_weight"):
        if payload.get("index_code") != "000300.SH":
            continue
        frame = pd.read_pickle(path)
        frame = frame[frame.trade_date.astype(str) < start]
        if len(frame):
            weight_frames.append(frame)
            sources.append(path)
    if not weight_frames:
        raise ValueError("No HS300 constituent snapshot predating start; cache it first")
    weights = pd.concat(weight_frames, ignore_index=True)
    snapshot = weights.trade_date.astype(str).max()
    codes = sorted(weights.loc[weights.trade_date.astype(str) == snapshot,
                               "con_code"].astype(str).unique())
    if assets < 2 or assets > len(codes):
        raise ValueError("Invalid fixed pool size")
    daily = dated_cache(root, "daily", ("ts_code", "trade_date", "open", "close", "pre_close", "vol"))
    basic = dated_cache(root, "daily_basic", ("ts_code", "trade_date", "pb", "turnover_rate",
                                            "volume_ratio", "circ_mv"))
    dates = sorted(d for d in daily if d <= end)
    past = [d for d in dates if d < start]
    evaluation = [d for d in dates if d >= start]
    calendar_dates = set()
    for _, path in cache_records(root, "trade_cal"):
        calendar = pd.read_pickle(path)
        opened = calendar.loc[pd.to_numeric(calendar.is_open) == 1, "cal_date"].astype(str)
        calendar_dates.update(d for d in opened if start <= d <= end)
        sources.append(path)
    if not calendar_dates or not calendar_dates.issubset(daily):
        raise ValueError("Cached calendar/quotes are incomplete; do not silently skip market days")
    if set(evaluation) != calendar_dates:
        raise ValueError("Daily cache and trading calendar disagree for the requested interval")
    if len(past) < 21 or not evaluation:
        raise ValueError("Need 21 cached warmup days before start and evaluation days")
    # Selection uses only the last available pre-start observation.
    initial = pd.read_pickle(daily[past[-1]])
    initial = initial[(initial.vol > 0) & (initial.close > 0)]
    codes = sorted(set(codes).intersection(initial.ts_code.astype(str)))
    if len(codes) < assets:
        raise ValueError("Insufficient tradable stocks at the pre-start date")
    selected = sorted(np.random.default_rng(pool_seed).choice(codes, assets, replace=False))
    frames = []
    previous_quote = {}
    previous_basic = {}
    suspension_dates = []
    for date in past[-21:] + evaluation:
        if date not in basic:
            raise ValueError(f"Missing daily_basic cache at {date}")
        quote = pd.read_pickle(daily[date])
        quote = quote[quote.ts_code.astype(str).isin(selected)].copy()
        basic_frame = pd.read_pickle(basic[date])
        basic_frame = basic_frame[basic_frame.ts_code.astype(str).isin(selected)].copy()
        if set(quote.trade_date.astype(str)) - {date}:
            raise ValueError(f"Cache contents disagree with date key {date}")
        if set(basic_frame.trade_date.astype(str)) - {date}:
            raise ValueError(f"Cache contents disagree with date key {date}")
        quote_rows, basic_rows = [], []
        for code in selected:
            q = quote[quote.ts_code.astype(str) == code]
            b = basic_frame[basic_frame.ts_code.astype(str) == code]
            quote_observed = len(q) == 1
            basic_observed = len(b) == 1
            if len(q) > 1 or len(b) > 1:
                raise ValueError(f"Duplicate fixed-pool observation for {code} at {date}")
            if not quote_observed and code not in previous_quote:
                raise ValueError(f"No prior quote for fixed-pool stock {code} at {date}")
            if not basic_observed and code not in previous_basic:
                raise ValueError(f"No prior factor data for fixed-pool stock {code} at {date}")
            if quote_observed and not basic_observed:
                raise ValueError(f"Factor data missing while quote is observed for fixed-pool stock "
                                 f"{code} at {date}; refresh daily_basic cache")
            if (not quote_observed or not basic_observed) and not allow_suspension_carry:
                missing = []
                if not quote_observed:
                    missing.append("daily")
                if not basic_observed:
                    missing.append("daily_basic")
                raise ValueError(f"Fixed pool has missing data at {date} for {code} "
                                 f"({', '.join(missing)}); use --allow-suspension-carry "
                                 "only after auditing the suspension status")
            if not quote_observed or not basic_observed:
                suspension_dates.append(dict(trade_date=date, ts_code=code,
                                             quote_observed=quote_observed,
                                             basic_observed=basic_observed))
            q_row = (q.iloc[0] if quote_observed else previous_quote[code]).copy()
            b_row = (b.iloc[0] if basic_observed else previous_basic[code]).copy()
            q_row["ts_code"] = code
            q_row["trade_date"] = date
            b_row["ts_code"] = code
            b_row["trade_date"] = date
            q_row["quote_observed"] = int(quote_observed)
            b_row["basic_observed"] = int(basic_observed)
            quote_rows.append(q_row)
            basic_rows.append(b_row)
            if quote_observed:
                previous_quote[code] = q_row
            if basic_observed:
                previous_basic[code] = b_row
        quote = pd.DataFrame(quote_rows)
        factors = pd.DataFrame(basic_rows)
        frame = quote.merge(factors[["ts_code", "trade_date", "pb", "turnover_rate",
                                     "volume_ratio", "circ_mv", "basic_observed"]],
                            on=["ts_code", "trade_date"], validate="one_to_one")
        required = ["open", "close", "pre_close", "vol", "pb", "turnover_rate",
                    "volume_ratio", "circ_mv"]
        if not np.isfinite(frame[required].to_numpy(float)).all():
            raise ValueError(f"Nonfinite fixed-pool observation at {date}")
        if (frame[["open", "close", "pre_close", "vol", "pb", "circ_mv"]] <= 0).any().any():
            raise ValueError(f"Nontradable/invalid fixed-pool observation at {date}; inspect, do not reselect")
        frames.append(frame)
        sources.extend([daily[date], basic[date]])
    panel = pd.concat(frames).sort_values(["trade_date", "ts_code"])
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    metadata = dict(snapshot=str(snapshot), selection_date=past[-1], start=start, end=end,
                    pool_seed=pool_seed, assets=selected, universe="initial HS300 constituents",
                    selection="pre-start tradability only; includes financial stocks; no future reselection",
                    suspension_carry=bool(allow_suspension_carry),
                    suspension_dates=suspension_dates,
                    features=FEATURES, panel_sha256=file_hash(output),
                    sources={str(p): file_hash(p) for p in sorted(set(sources))})
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@dataclass
class PaperPanel:
    dates: np.ndarray
    codes: list
    features: np.ndarray
    returns: np.ndarray
    corporate_break: np.ndarray
    tradable: np.ndarray | None = None

    def __post_init__(self):
        if self.tradable is None:
            self.tradable = np.ones((len(self.dates), len(self.codes)), dtype=bool)

    @classmethod
    def load(cls, path, start, end):
        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if file_hash(path) != metadata["panel_sha256"]:
            raise ValueError("Prepared panel checksum mismatch")
        if metadata["snapshot"] >= start or metadata["selection_date"] >= start:
            raise ValueError("Asset selection is not strictly earlier than evaluation")
        df = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
        codes = metadata["assets"]
        def matrix(name):
            return df.pivot(index="trade_date", columns="ts_code", values=name).sort_index()[codes]
        close, previous = matrix("close"), matrix("pre_close")
        quote_observed = matrix("quote_observed").astype(bool)
        real_return = (close / previous - 1).where(quote_observed, 0.0)
        signals = [real_return.rolling(20).mean().shift(1),
                   -real_return.rolling(5).mean().shift(1),
                   -real_return.rolling(20).std(ddof=0).shift(1),
                   -np.log(matrix("pb")).shift(1),
                   np.log1p(matrix("turnover_rate")).shift(1),
                   np.log(matrix("circ_mv")).shift(1) - abs(matrix("volume_ratio").shift(1) - 1)]
        mask = (close.index >= start) & (close.index <= end)
        dates = close.index[mask].to_numpy()
        features = np.zeros((len(dates), len(codes) + 1, len(FEATURES)))
        features[:, :, 0] = 1
        features[:, 0, 1] = 1
        for col, signal in enumerate(signals, 2):
            values = signal.loc[dates].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("Incomplete lagged factors; expand pre-start cache")
            sd = values.std(axis=1, keepdims=True)
            features[:, 1:, col] = np.clip((values - values.mean(axis=1, keepdims=True)) /
                                           np.where(sd > 0, sd, 1), -3, 3) / 3
        features[:, :, 8] = np.tanh(real_return.mean(axis=1).rolling(20).mean().shift(1)
                                             .loc[dates].to_numpy()[:, None] / .02)
        # A reference-price jump flags corporate actions; it is NOT a total-return adjustment.
        breaks = ((abs(previous / close.shift(1) - 1) > 1e-4) & quote_observed).loc[dates].to_numpy()
        returns = np.c_[np.zeros(len(dates)), real_return.loc[dates].to_numpy()]
        if not len(dates) or not np.isfinite(features).all() or not np.isfinite(returns).all():
            raise ValueError("Invalid prepared panel")
        tradable = np.c_[np.ones(len(dates), dtype=bool), quote_observed.loc[dates].to_numpy()]
        return cls(dates, ["CASH", *codes], features, returns, breaks, tradable)


@dataclass
class PaperMarket:
    panel: PaperPanel
    semi: bool
    factor_strength: float = 2.5
    return_bound: float = .08

    def __post_init__(self):
        self.regimes = _regime_labels(len(self.panel.dates), 0) if self.semi else ["real_market"] * len(self.panel.dates)

    def regime(self, day):
        return self.regimes[day]

    def draw_returns(self, day, features, rng):
        coefs, mean, scale = _semi_synthetic_regime_parameters(self.regime(day))
        signal = np.zeros(features.shape[:-1])
        for name, coef in coefs.items():
            signal += coef * features[..., FEATURES.index(name)]
        shape = features.shape[:-1]
        common = rng.normal(0, .006 * scale, size=(*shape[:-1], 1))
        noise = rng.normal(0, .009 * scale, size=shape)
        returns = self.return_bound * np.tanh((mean + self.factor_strength * signal + common + noise)
                                             / self.return_bound)
        returns[..., 0] = 0
        return returns

    def execution_returns(self, seed):
        if not self.semi:
            return self.panel.returns.copy()
        rng = np.random.default_rng(seed)
        result = np.stack([self.draw_returns(day, features, rng)
                           for day, features in enumerate(self.panel.features)])
        result[~self.panel.tradable] = 0.0
        return result
