"""Narrows the historical tickets down to candidates whose metadata matches the new ticket.

Rules:
* A filter is applied only when the incoming ticket has a known value for it.
  Unknown values are skipped, never guessed.
* A historical ticket with an unknown value never matches a known query value.
* The incoming ticket itself, and ticket versions that were replaced or
  removed (is_active = False), are always excluded.
* In relaxed mode the filter set is loosened one field at a time
  (filters.relax_order), producing a sequence of FilterLevels from strict to
  loose. The recommendation service walks through them until it has enough
  results. Fields not in relax_order (IspId by default) are never dropped.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RE_INTEGER_FLOAT = re.compile(r"^(-?\d+)\.0+$")


@dataclass
class FilterLevel:
    mask: np.ndarray
    applied: list[str]
    relaxed: list[str]
    candidate_count: int


@dataclass
class FilterPlan:
    levels: list[FilterLevel]
    query_values: dict[str, str]                  # field -> value used for filtering
    skipped: dict[str, str]                       # field -> reason it was not applied
    warnings: list[str] = field(default_factory=list)


class MetadataFilter:
    def __init__(self, metadata: pd.DataFrame, filter_cfg: dict, null_tokens: set[str]) -> None:
        self.cfg = filter_cfg
        self.mode = filter_cfg.get("mode", "relaxed")
        self.unknown = {str(v).strip().lower() for v in set(null_tokens) | set(filter_cfg.get("unknown_values", []))}
        self.casefold = set(filter_cfg.get("case_insensitive_fields", []))
        self.fields = [f for f in filter_cfg.get("fields", []) if f in metadata.columns]
        self.unavailable = [f for f in filter_cfg.get("fields", []) if f not in metadata.columns]
        if self.unavailable:
            log.warning("Filter fields not present in the index: %s", self.unavailable)

        self.ticket_numbers = metadata["TicketNumber"].to_numpy(dtype=str)
        self.active = metadata["is_active"].to_numpy(dtype=bool)
        self._codes: dict[str, np.ndarray] = {}
        self._vocab: dict[str, dict[str, int]] = {}
        for name in self.fields:
            categories = pd.Categorical(self._normalize_series(name, metadata[name]))
            self._codes[name] = categories.codes.astype(np.int32)        # -1 = unknown
            self._vocab[name] = {value: i for i, value in enumerate(categories.categories)}

    # ---- value normalisation (identical for history and query) ----------------
    def _normalize_series(self, name: str, series: pd.Series) -> pd.Series:
        s = series.astype("string").str.strip().str.replace(RE_INTEGER_FLOAT.pattern, r"\1", regex=True)
        s = s.mask(s.str.lower().isin(self.unknown))
        return s.str.lower() if name in self.casefold else s

    def normalize(self, name: str, value) -> str | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = RE_INTEGER_FLOAT.sub(r"\1", str(value).strip())
        if text.lower() in self.unknown:
            return None
        return text.lower() if name in self.casefold else text

    # ---- masks ---------------------------------------------------------------
    def _field_mask(self, name: str, key: str) -> np.ndarray:
        code = self._vocab[name].get(key)
        if code is None:
            return np.zeros(len(self.active), dtype=bool)
        return self._codes[name] == code

    def base_mask(self, exclude_ticket: str | None) -> np.ndarray:
        mask = self.active.copy()
        if exclude_ticket:
            mask &= self.ticket_numbers != str(exclude_ticket)
        return mask

    def plan(self, query: dict, exclude_ticket: str | None = None, mode: str | None = None) -> FilterPlan:
        """Filter levels for this query; `mode` overrides filters.mode for one call."""
        mode = mode or self.mode
        base = self.base_mask(exclude_ticket)
        query_values, keys, skipped, warnings = {}, {}, {}, []
        for name in self.cfg.get("fields", []):
            if name in self.unavailable:
                skipped[name] = "field not available in historical data"
                continue
            key = self.normalize(name, query.get(name))
            if key is None:
                skipped[name] = "unknown in incoming ticket"
                continue
            if key not in self._vocab[name]:
                warnings.append(f"{name}={query.get(name)!r} never occurs in the historical data")
            keys[name], query_values[name] = key, str(query.get(name))

        field_masks = {name: self._field_mask(name, key) for name, key in keys.items()}
        active_fields = list(keys)
        relax_queue = [f for f in self.cfg.get("relax_order", []) if f in keys] if mode == "relaxed" else []
        relaxed: list[str] = []
        levels: list[FilterLevel] = []
        while True:
            mask = base.copy()
            for name in active_fields:
                mask &= field_masks[name]
            levels.append(FilterLevel(mask, list(active_fields), list(relaxed), int(mask.sum())))
            if not relax_queue:
                break
            dropped = relax_queue.pop(0)
            active_fields.remove(dropped)
            relaxed.append(dropped)
        return FilterPlan(levels, query_values, skipped, warnings)
