"""Read ticket and article CSV files in chunks and map their columns to canonical names.

Each file is inspected first: header or not, which columns are present, and
which required columns are missing. Missing required columns stop the run
with a clear error instead of being silently ignored.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

from config import PROJECT_ROOT, Config

log = logging.getLogger(__name__)


class SchemaError(ValueError):
    """A CSV file does not contain the columns the pipeline needs."""


@dataclass
class CsvSchema:
    file: Path
    has_header: bool
    mapping: dict[str, str]                 # source column -> canonical column
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    unmapped_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file": str(self.file),
            "has_header": self.has_header,
            "mapped_columns": self.mapping,
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "ignored_columns": len(self.unmapped_columns),
        }


def _key(name: str) -> str:
    """Comparison key for column names: lowercase, letters and digits only."""
    return re.sub(r"[^0-9a-z]", "", str(name).replace("﻿", "").lower())


def resolve_files(patterns: list[str] | str) -> list[Path]:
    if isinstance(patterns, str):
        patterns = [patterns]
    files: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern)
        if path.is_absolute():
            files.update(path.parent.glob(path.name))
        else:
            files.update(PROJECT_ROOT.glob(pattern))
    return sorted(f for f in files if f.is_file())


def _alias_lookup(section: dict) -> dict[str, str]:
    """Map every known name (canonical and aliases) to its canonical name."""
    lookup = {_key(name): name for name in section["keep_columns"]}
    for canonical, aliases in (section.get("aliases") or {}).items():
        lookup[_key(canonical)] = canonical
        for alias in aliases:
            lookup.setdefault(_key(alias), canonical)
    return lookup


def inspect_schema(file: Path, section: dict, csv_cfg: dict) -> CsvSchema:
    with open(file, encoding=csv_cfg["encoding"], errors="replace", newline="") as handle:
        first_row = next(csv.reader(handle, delimiter=csv_cfg["delimiter"]), [])
    first_row = [col.strip().strip('"') for col in first_row]

    lookup = _alias_lookup(section)
    recognised = sum(1 for col in first_row if _key(col) in lookup)
    has_header = section.get("has_header", "auto")
    if has_header == "auto":
        has_header = recognised >= 2

    if has_header:
        source_columns = first_row
    else:
        source_columns = list(section["columns"])
        if len(first_row) != len(source_columns):
            raise SchemaError(
                f"{file.name}: no header row and {len(first_row)} fields per row, but "
                f"config lists {len(source_columns)} columns. Update the 'columns' list."
            )

    keep = set(section["keep_columns"])
    mapping: dict[str, str] = {}
    for col in source_columns:
        canonical = lookup.get(_key(col))
        if canonical in keep and canonical not in mapping.values():
            mapping[col] = canonical

    present = set(mapping.values())
    required = list(section["required_columns"])
    return CsvSchema(
        file=file,
        has_header=bool(has_header),
        mapping=mapping,
        missing_required=[c for c in required if c not in present],
        missing_optional=[c for c in section["keep_columns"] if c not in present and c not in required],
        unmapped_columns=[c for c in source_columns if c not in mapping],
    )


def iter_csv_chunks(kind: str, cfg: Config, files: list[Path] | None = None) -> Iterator[tuple[CsvSchema, pd.DataFrame]]:
    """Yield (schema, chunk) for every chunk of every file of `kind` ("tickets" or "articles").

    Chunks contain every column in keep_columns (missing optional ones as None),
    all values as strings, nothing parsed or converted yet.
    """
    section, csv_cfg = cfg[kind], cfg["csv"]
    files = files if files is not None else resolve_files(cfg["paths"][f"{kind}_csv"])
    if not files:
        raise FileNotFoundError(f"No {kind} CSV files match {cfg['paths'][f'{kind}_csv']}")

    for file in files:
        schema = inspect_schema(file, section, csv_cfg)
        if schema.missing_required:
            raise SchemaError(f"{file.name}: required {kind} columns missing: {schema.missing_required}")
        if schema.missing_optional:
            log.warning("%s: optional columns not found: %s", file.name, schema.missing_optional)
        log.info("Reading %s (%.1f MB, header=%s, %d columns used)",
                 file.name, file.stat().st_size / 1e6, schema.has_header, len(schema.mapping))

        reader = pd.read_csv(
            file,
            sep=csv_cfg["delimiter"],
            header=0 if schema.has_header else None,
            names=None if schema.has_header else section["columns"],
            usecols=list(schema.mapping),
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding=csv_cfg["encoding"],
            encoding_errors="replace",
            on_bad_lines="warn",
            chunksize=int(csv_cfg["chunk_size"]),
        )
        for chunk in reader:
            chunk = chunk.rename(columns=schema.mapping)
            for col in section["keep_columns"]:
                if col not in chunk.columns:
                    chunk[col] = None
            yield schema, chunk[section["keep_columns"]]
