"""Coordinates the full retrieval pipeline for one incoming ticket.

    new ticket -> validate + normalise -> ticket_text -> query embedding
               -> metadata filter levels (strict ... relaxed)
               -> exact cosine ranking of the eligible candidates
               -> top-k similar resolved tickets
               -> their linked articles, ranked -> JSON response

The index is loaded once (RecommendationService.__init__) and reused for
every query; nothing is rebuilt at query time.

similarity_score is the cosine similarity between the two ticket embeddings
(-1..1, in practice 0..1). similarity_percent = cosine * 100. Neither is a
probability that the historical resolution will work.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from article_retriever import ArticleRetriever
from config import Config, load_config
from embedding_service import EmbeddingService
from metadata_filter import MetadataFilter
from ticket_preprocessor import build_ticket_text, is_missing, preprocessing_fingerprint
from vector_store import VectorStore

log = logging.getLogger(__name__)

SIMILARITY_NOTE = ("similarity_score = cosine similarity of the ticket embeddings; similarity_percent = "
                   "cosine x 100. It measures textual similarity, not the probability that a resolution works.")
TITLE_KEYS = ("Title_ticket", "Title")
BODY_KEYS = ("BodyText_ticket", "BodyText", "Description")


class InvalidTicketError(ValueError):
    pass


def _first_present(ticket: dict, keys: tuple[str, ...]):
    for key in keys:
        if not is_missing(ticket.get(key)):
            return ticket[key]
    return None


class RecommendationService:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.store = VectorStore(self.cfg)
        self._check_build_settings()
        self.embedder = EmbeddingService(self.cfg["embedding"])
        self.embedder.check_matches_probe(self.store.manifest.get("probe_embedding"),
                                          strict=self.cfg["embedding"].get("strict_model_check", True))
        self.filter = MetadataFilter(self.store.metadata, self.cfg["filters"], self.cfg.null_tokens)
        self.articles = ArticleRetriever(self.cfg)

    def _check_build_settings(self) -> None:
        manifest = self.store.manifest
        built_model = manifest["model"]["model_name"]
        if built_model != self.cfg["embedding"]["model_name"]:
            raise RuntimeError(f"Index was built with '{built_model}' but config uses "
                               f"'{self.cfg['embedding']['model_name']}'. Rebuild or fix config.yaml.")
        if manifest["preprocessing"]["fingerprint"] != preprocessing_fingerprint(self.cfg["text"]):
            raise RuntimeError("The 'text' settings changed since the index was built; "
                               "run python build_index.py to rebuild.")

    # ---- input ---------------------------------------------------------------
    def prepare_ticket(self, new_ticket: dict) -> tuple[dict, str, list[str]]:
        """Validate the incoming ticket, map aliases and build its ticket_text."""
        if not isinstance(new_ticket, dict):
            raise InvalidTicketError("A ticket must be a dict of field -> value")
        ticket, warnings = dict(new_ticket), []

        for alias, target in (self.cfg["filters"].get("input_aliases") or {}).items():
            if alias not in ticket:
                continue
            value = ticket.pop(alias)
            if is_missing(ticket.get(target)):
                ticket[target] = value
                warnings.append(f"'{alias}' was used as {target} (the source data has no '{alias}' column).")
            elif str(ticket[target]).strip().lower() != str(value).strip().lower():
                warnings.append(f"'{alias}'={value!r} conflicts with {target}={ticket[target]!r}; {target} is used.")

        title, body = _first_present(ticket, TITLE_KEYS), _first_present(ticket, BODY_KEYS)
        ticket_text = build_ticket_text(title, body, self.cfg["text"], ticket.get("AdditionalInformation"))
        if not ticket_text:
            raise InvalidTicketError("The ticket needs a title or a description.")
        return ticket, ticket_text, warnings

    # ---- step 1: similar tickets -----------------------------------------------
    def find_similar_tickets(self, new_ticket: dict, top_k: int | None = None, **overrides) -> dict:
        result, _ = self._find_similar(new_ticket, top_k, **overrides)
        return result

    def _find_similar(self, new_ticket: dict, top_k: int | None, min_similarity: float | None = None,
                      filter_mode: str | None = None) -> tuple[dict, np.ndarray]:
        """`min_similarity` and `filter_mode` override config.yaml for this one query (used by the UI)."""
        retrieval_cfg, filter_cfg = self.cfg["retrieval"], self.cfg["filters"]
        top_k = int(top_k or retrieval_cfg.get("top_k", 5))
        if min_similarity is None:
            min_similarity = float(retrieval_cfg.get("min_similarity", 0.0))
        min_candidates = int(filter_cfg.get("min_candidates", 1))

        ticket, ticket_text, warnings = self.prepare_ticket(new_ticket)
        query_id = None if is_missing(ticket.get("TicketNumber")) else str(ticket["TicketNumber"]).strip()
        query_vector = self.embedder.encode_one(ticket_text)
        plan = self.filter.plan(ticket, exclude_ticket=query_id, mode=filter_mode)
        warnings += plan.warnings

        prefer_stricter = filter_cfg.get("prefer_stricter_matches", True)
        chosen: list[dict] = []
        seen: set[str] = set()
        trace = []
        for level_no, level in enumerate(plan.levels):
            rows, scores, method = self.store.search(query_vector, level.mask, top_k + len(seen))
            added = 0
            for row, score in zip(rows, scores):
                if score < min_similarity:
                    break
                number = self.filter.ticket_numbers[row]
                if number in seen:
                    continue
                seen.add(number)
                chosen.append({"TicketNumber": number, "score": float(score), "level": level_no})
                added += 1
            trace.append({"filters": level.applied, "relaxed": level.relaxed, "candidates": level.candidate_count,
                          "search_method": method, "tickets_added": added})
            # A full list from stricter levels cannot be improved by relaxing when
            # stricter matches rank first; otherwise relax until the pool is large enough.
            if len(chosen) >= top_k and (prefer_stricter or level.candidate_count >= min_candidates):
                break

        if not prefer_stricter:
            chosen.sort(key=lambda c: -c["score"])
        chosen = chosen[:top_k]

        used_level = plan.levels[max([c["level"] for c in chosen], default=len(trace) - 1)]
        tickets = self._ticket_records(chosen, plan)
        status, explanation = self._status(len(tickets), top_k, min_similarity, trace)
        if explanation:
            warnings.append(explanation)

        result = {
            "query_ticket": query_id,
            "retrieval_status": status,
            "metadata_filters_applied": used_level.applied,
            "filters_relaxed": used_level.relaxed,
            "filters_skipped": plan.skipped,
            "query_filter_values": plan.query_values,
            "min_similarity": min_similarity,
            "similar_tickets": tickets,
            "filter_trace": trace,
            "warnings": warnings,
        }
        return result, query_vector

    def _ticket_records(self, chosen: list[dict], plan) -> list[dict]:
        if not chosen:
            return []
        fields = self.cfg["retrieval"]["output_fields"]
        details = self.store.ticket_details([c["TicketNumber"] for c in chosen], columns=fields)
        details = details.astype(object).where(details.notna(), None).set_index("TicketNumber", drop=False)
        records = []
        for c in chosen:
            record = details.loc[c["TicketNumber"]].to_dict()
            record["similarity_score"] = round(c["score"], 4)
            record["similarity_percent"] = round(c["score"] * 100, 2)
            record["matched_filters"] = plan.levels[c["level"]].applied
            records.append(record)
        return records

    @staticmethod
    def _status(found: int, top_k: int, min_similarity: float, trace: list[dict]) -> tuple[str, str | None]:
        if found >= top_k:
            return "success", None
        searched = trace[-1]["candidates"] if trace else 0
        if found == 0:
            if searched == 0:
                return "no_results", "No eligible historical tickets match the metadata, even after relaxing filters."
            return "no_results", (f"No historical ticket reached the minimum similarity of {min_similarity} "
                                  f"({searched:,} candidates compared).")
        return "partial", (f"Only {found} of {top_k} tickets reached the minimum similarity of {min_similarity}; "
                           f"weaker matches are not returned to fill the list.")

    # ---- step 2: articles ------------------------------------------------------
    def get_relevant_articles(self, similar_tickets: list[dict]):
        return self.articles.get_relevant_articles(similar_tickets)

    # ---- full pipeline ---------------------------------------------------------
    def recommend_resolution(self, new_ticket: dict, top_k: int | None = None, **overrides) -> dict:
        started = time.time()
        try:
            result, _ = self._find_similar(new_ticket, top_k, **overrides)
        except InvalidTicketError as exc:
            return {"query_ticket": (new_ticket or {}).get("TicketNumber") if isinstance(new_ticket, dict) else None,
                    "retrieval_status": "invalid_input", "similar_tickets": [], "warnings": [str(exc)]}

        by_ticket, ranking, article_warnings = self.get_relevant_articles(result["similar_tickets"])
        for ticket in result["similar_tickets"]:
            ticket["articles"] = by_ticket.get(ticket["TicketNumber"], [])
            if self.articles.available and not ticket["articles"]:
                ticket["articles_note"] = "No linked resolution article found for this ticket."

        manifest = self.store.manifest
        response = {
            "query_ticket": result["query_ticket"],
            "retrieval_status": result["retrieval_status"],
            "similarity_metric": SIMILARITY_NOTE,
            "metadata_filters_applied": result["metadata_filters_applied"],
            "filters_relaxed": result["filters_relaxed"],
            "filters_skipped": result["filters_skipped"],
            "similar_tickets": result["similar_tickets"],
            "article_ranking": ranking,
            "filter_trace": result["filter_trace"],
            "warnings": result["warnings"] + article_warnings,
            "index": {"index_version": manifest.get("index_version"), "dataset_version": manifest.get("dataset_version"),
                      "model": manifest["model"]["model_name"]},
            "elapsed_ms": round((time.time() - started) * 1000),
        }
        log.info("Recommendation for %s: %s, %d tickets, %d articles",
                 response["query_ticket"], response["retrieval_status"], len(response["similar_tickets"]),
                 len(ranking), extra={"data": {k: response[k] for k in
                                              ("query_ticket", "retrieval_status", "metadata_filters_applied",
                                               "filters_relaxed", "filter_trace", "elapsed_ms")}})
        return response


# ---- module-level entry points (one shared service per process) ----------------
_service: RecommendationService | None = None


def get_service(cfg: Config | None = None) -> RecommendationService:
    global _service
    if _service is None:
        _service = RecommendationService(cfg)
    return _service


def find_similar_tickets(new_ticket: dict, top_k: int = 5) -> dict:
    return get_service().find_similar_tickets(new_ticket, top_k)


def get_relevant_articles(similar_tickets: list[dict]):
    return get_service().get_relevant_articles(similar_tickets)


def recommend_resolution(new_ticket: dict, top_k: int = 5) -> dict:
    return get_service().recommend_resolution(new_ticket, top_k)
