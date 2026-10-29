"""Row-level cleaning, validation and de-duplication of tickets and articles.

Every excluded row is counted under a reason, and the counts end up in
cleaning_report.json / article_report.json, so it is always traceable why a
ticket is (not) in the index.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import Config
from data_loader import columns_to_read, iter_csv_chunks
from ticket_preprocessor import build_ticket_text, clean_markup, mask_personal_data, text_hash

log = logging.getLogger(__name__)

TICKET_TEXT_COLUMNS = ["Title_ticket", "BodyText_ticket", "AdditionalInformation"]
ARTICLE_TEXT_COLUMNS = ["Title_article", "BodyText_article"]


@dataclass
class CleanResult:
    data: pd.DataFrame
    report: dict
    seen_ids: set[str]          # every valid TicketNumber/Id in the input, kept or not


class _Exclusions:
    """Counts excluded rows per reason and keeps a few example ids."""

    def __init__(self, id_column: str, max_samples: int) -> None:
        self.id_column = id_column
        self.max_samples = max_samples
        self.counts: Counter = Counter()
        self.samples: dict[str, list[str]] = defaultdict(list)

    def drop(self, df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
        if not mask.any():
            return df
        self.counts[reason] += int(mask.sum())
        room = self.max_samples - len(self.samples[reason])
        if room > 0:
            ids = df.loc[mask, self.id_column].dropna().astype(str).head(room)
            self.samples[reason].extend(ids)
        return df[~mask]

    def to_dict(self) -> dict:
        return {"excluded": dict(self.counts.most_common()), "excluded_samples": dict(self.samples)}


def normalize_ids(series: pd.Series, null_tokens: set[str]) -> pd.Series:
    """Trim whitespace and turn null tokens into None. "11.0" becomes "11";
    every other value (including leading zeros) is kept exactly."""
    s = series.astype("string").str.strip()
    s = s.mask(s.isin(null_tokens))
    s = s.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    return s.astype(object).where(s.notna(), None)


def _count_missing(df: pd.DataFrame, counter: Counter) -> None:
    for col in df.columns:
        counter[col] += int(df[col].isna().sum() + (df[col] == "").sum())


def _dedupe(df: pd.DataFrame, key: str, order_column: str | None) -> tuple[pd.DataFrame, int]:
    """Keep one row per key: the one with the latest order_column value."""
    before = len(df)
    if order_column and order_column in df.columns:
        df = df.sort_values(order_column, na_position="first", kind="stable")
    df = df.drop_duplicates(subset=[key], keep="last")
    return df, before - len(df)


# -----------------------------------------------------------------------------
# Tickets
# -----------------------------------------------------------------------------
def clean_tickets(cfg: Config, files: list[Path] | None = None) -> CleanResult:
    rules, text_cfg = cfg["cleaning"], cfg["text"]
    null_tokens = cfg.null_tokens
    eligibility = rules.get("eligibility", {})
    allowed_states = set(eligibility.get("allowed_states") or [])
    excluded_substates = set(eligibility.get("excluded_substates") or [])
    patterns = rules.get("drop_title_patterns", [])
    noise_title = re.compile("|".join(f"(?:{p})" for p in patterns), re.I) if patterns else None
    min_chars = int(rules.get("min_text_chars", 15))

    exclusions = _Exclusions("TicketNumber", int(rules.get("report_samples", 5)))
    missing = Counter()
    schemas, parts, seen_ids = {}, [], set()
    input_rows = 0

    for schema, chunk in iter_csv_chunks("tickets", cfg, files):
        schemas[str(schema.file)] = schema.to_dict()
        input_rows += len(chunk)

        meta_cols = [c for c in chunk.columns if c not in TICKET_TEXT_COLUMNS]
        for col in meta_cols:
            chunk[col] = normalize_ids(chunk[col], null_tokens)
        _count_missing(chunk, missing)

        df = exclusions.drop(chunk, chunk["TicketNumber"].isna(), "missing_ticket_number")
        seen_ids.update(df["TicketNumber"])

        if allowed_states:
            bad_state = ~df["State"].isin(allowed_states)
            for state in df.loc[bad_state, "State"].fillna("<missing>").unique():
                df = exclusions.drop(df, df["State"].fillna("<missing>") == state, f"ineligible_state:{state}")
        if excluded_substates:
            df = exclusions.drop(df, df["SubState"].isin(excluded_substates), "excluded_substate")
        if eligibility.get("require_close_date", False):
            df = exclusions.drop(df, df["CloseDate"].isna(), "missing_close_date")
        for field in rules.get("mandatory_fields", []):
            if field != "TicketNumber":
                df = exclusions.drop(df, df[field].isna(), f"missing_{field}")
        if df.empty:
            continue

        df = df.copy()
        for col in TICKET_TEXT_COLUMNS:
            if col in df.columns:
                df[col] = [clean_markup(v) for v in df[col]]
        df = exclusions.drop(df, (df["Title_ticket"] == "") & (df["BodyText_ticket"] == ""),
                             "empty_title_and_description")
        if noise_title is not None:
            noise = df["Title_ticket"].map(lambda title: bool(noise_title.search(title)))
            df = exclusions.drop(df, noise, "test_or_noise_title")
        if df.empty:
            continue

        extra = df["AdditionalInformation"] if "AdditionalInformation" in df.columns else [None] * len(df)
        df["ticket_text"] = [
            build_ticket_text(t, b, text_cfg, a) for t, b, a in zip(df["Title_ticket"], df["BodyText_ticket"], extra)
        ]
        text_len = df["ticket_text"].str.replace(r"^(Title|Description): ", "", regex=True).str.len()
        df = exclusions.drop(df, text_len < min_chars, "text_too_short")
        parts.append(df)
        log.info("Tickets read: %s | kept so far: %s", f"{input_rows:,}", f"{sum(map(len, parts)):,}")

    tickets = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns_to_read("tickets", cfg))
    tickets, duplicate_ids = _dedupe(tickets, "TicketNumber", rules.get("dedupe_order_column"))

    duplicate_text = 0
    if rules.get("drop_identical_text", False) and not tickets.empty:
        tickets, duplicate_text = _dedupe(tickets, "ticket_text", rules.get("dedupe_order_column"))

    tickets["text_hash"] = [text_hash(t) for t in tickets["ticket_text"]]
    tickets = tickets.sort_values("TicketNumber", kind="stable").reset_index(drop=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": list(schemas.values()),
        "input_rows": input_rows,
        "output_rows": len(tickets),
        "duplicates_removed": {"same_ticket_number": duplicate_ids, "identical_ticket_text": duplicate_text},
        "missing_values_in_input": dict(missing),
        **exclusions.to_dict(),
        "rules": {"cleaning": rules, "text": text_cfg},
    }
    log.info("Ticket cleaning done: %s in -> %s out", f"{input_rows:,}", f"{len(tickets):,}")
    return CleanResult(tickets, report, seen_ids)


# -----------------------------------------------------------------------------
# Articles
# -----------------------------------------------------------------------------
def clean_articles(cfg: Config, ticket_ids: set[str] | None = None, files: list[Path] | None = None) -> CleanResult:
    acfg = cfg["articles"]
    null_tokens = cfg.null_tokens
    excluded_types = {str(t) for t in acfg.get("exclude_article_type_ids") or []}
    exclusions = _Exclusions("Id", int(cfg["cleaning"].get("report_samples", 5)))
    missing = Counter()
    schemas, parts, seen_ids = {}, [], set()
    input_rows = 0

    for schema, chunk in iter_csv_chunks("articles", cfg, files):
        schemas[str(schema.file)] = schema.to_dict()
        input_rows += len(chunk)

        for col in chunk.columns:
            if col not in ARTICLE_TEXT_COLUMNS:
                chunk[col] = normalize_ids(chunk[col], null_tokens)
        _count_missing(chunk, missing)

        df = exclusions.drop(chunk, chunk["Id"].isna(), "missing_article_id")
        df = exclusions.drop(df, df["TicketNumber"].isna(), "missing_ticket_number")
        seen_ids.update(df["Id"])
        if excluded_types:
            df = exclusions.drop(df, df["ArticleTypeId"].isin(excluded_types), "excluded_article_type")
        if ticket_ids is not None:
            df = exclusions.drop(df, ~df["TicketNumber"].isin(ticket_ids), "ticket_not_in_index")
        if df.empty:
            continue

        df = df.copy()
        for col in ARTICLE_TEXT_COLUMNS:
            df[col] = [clean_markup(v) for v in df[col]]
            if acfg.get("mask_pii", False):
                df[col] = [mask_personal_data(v) for v in df[col]]
        df = exclusions.drop(df, (df["Title_article"] == "") & (df["BodyText_article"] == ""), "empty_article")
        parts.append(df)
        log.info("Articles read: %s | kept so far: %s", f"{input_rows:,}", f"{sum(map(len, parts)):,}")

    articles = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns_to_read("articles", cfg))
    articles, duplicate_ids = _dedupe(articles, "Id", "CreateDate")
    articles["content_hash"] = [
        text_hash(f"{t}\n{b}\n{ty}") for t, b, ty in
        zip(articles["Title_article"], articles["BodyText_article"], articles["ArticleTypeId"])
    ]
    # Sorted by TicketNumber so a lookup of a few tickets reads only a few row groups.
    articles = articles.sort_values(["TicketNumber", "CreateDate", "Id"], kind="stable").reset_index(drop=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_files": list(schemas.values()),
        "input_rows": input_rows,
        "output_rows": len(articles),
        "tickets_with_articles": int(articles["TicketNumber"].nunique()),
        "duplicates_removed": {"same_article_id": duplicate_ids},
        "missing_values_in_input": dict(missing),
        **exclusions.to_dict(),
    }
    log.info("Article cleaning done: %s in -> %s out", f"{input_rows:,}", f"{len(articles):,}")
    return CleanResult(articles, report, seen_ids)
