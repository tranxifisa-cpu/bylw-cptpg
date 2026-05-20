from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def progress(iterable: Iterable[T], *, desc: str, total: int | None = None) -> Iterator[T]:
    try:
        from tqdm import tqdm
    except Exception:
        yield from iterable
        return
    yield from tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
