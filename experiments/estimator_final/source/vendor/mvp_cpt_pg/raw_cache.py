from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .utils import ensure_dir, stable_hash


@dataclass
class CacheRecord:
    data_path: Path
    meta_path: Path


class DataFrameCache:
    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(root)

    def _record(self, namespace: str, payload: dict[str, Any]) -> CacheRecord:
        ns_dir = ensure_dir(self.root / namespace)
        digest = stable_hash(payload)
        return CacheRecord(
            data_path=ns_dir / f"{digest}.pkl",
            meta_path=ns_dir / f"{digest}.json",
        )

    def get_or_fetch(
        self,
        namespace: str,
        payload: dict[str, Any],
        fetcher: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        record = self._record(namespace, payload)
        if record.data_path.exists():
            return pd.read_pickle(record.data_path)
        df = fetcher()
        df.to_pickle(record.data_path)
        record.meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return df
