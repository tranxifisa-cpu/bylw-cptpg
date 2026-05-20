from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def maybe_datetime(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    text = str(value).strip()
    if not text:
        return pd.NaT
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return pd.to_datetime(text, errors="coerce")


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std < 1e-12:
        return pd.Series(np.zeros(len(values)), index=series.index, dtype=float)
    return ((values - mean) / std).fillna(0.0)


def normalize_weights(weights: pd.Series) -> pd.Series:
    clipped = weights.clip(lower=0.0)
    total = float(clipped.sum())
    if total <= 0:
        return pd.Series(np.zeros(len(weights)), index=weights.index, dtype=float)
    return clipped / total


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("no json object found")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError("no balanced json object found")


def get_env_var(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None
    locations = (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for root, path in locations:
        try:
            with winreg.OpenKey(root, path) as key:
                for name in names:
                    try:
                        value, _ = winreg.QueryValueEx(key, name)
                    except FileNotFoundError:
                        continue
                    if value:
                        return str(value)
        except OSError:
            continue
    return None
