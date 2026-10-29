"""Atomic file writes so an interrupted run never leaves a half-written artifact."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _tmp_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.tmp")


class AtomicBatch:
    """Write several files to temporary paths and move them into place together.

    Used where files must stay consistent with each other (FAISS index,
    metadata, manifest): nothing is replaced until every file was written.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[Path, Path]] = []

    def stage(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_path(target)
        self._pending.append((tmp, target))
        return tmp

    def commit(self) -> None:
        for tmp, target in self._pending:
            os.replace(tmp, target)
        self._pending.clear()

    def rollback(self) -> None:
        for tmp, _ in self._pending:
            tmp.unlink(missing_ok=True)
        self._pending.clear()

    def __enter__(self) -> "AtomicBatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


def save_json(obj, path: Path) -> None:
    with AtomicBatch() as batch:
        write_json(obj, batch.stage(path))


def save_parquet(df: pd.DataFrame, path: Path, row_group_size: int = 50_000) -> None:
    with AtomicBatch() as batch:
        write_parquet(df, batch.stage(path), row_group_size)


def save_npy(array: np.ndarray, path: Path) -> None:
    with AtomicBatch() as batch:
        write_npy(array, batch.stage(path))


def write_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=str)


def write_parquet(df: pd.DataFrame, path: Path, row_group_size: int = 50_000) -> None:
    # Small row groups + a sorted key column let readers skip most of the file
    # when they filter on that key (used for TicketNumber lookups).
    df.to_parquet(path, index=False, engine="pyarrow", row_group_size=row_group_size)


def write_npy(array: np.ndarray, path: Path) -> None:
    with open(path, "wb") as handle:  # a file handle stops numpy appending ".npy"
        np.save(handle, array)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
