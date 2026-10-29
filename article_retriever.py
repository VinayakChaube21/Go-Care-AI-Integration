"""Fetches the resolution articles linked to the similar tickets and ranks them.

Articles live in cleaned_articles.parquet, sorted by TicketNumber. Lookups
filter on TicketNumber, so pyarrow reads only the row groups that can contain
those tickets: the sorted file works as the ticket -> article index.

Ranking (article_ranking.mode):
  source_ticket : relevance = similarity of the ticket the article belongs to
  article_text  : relevance = cosine(incoming ticket, article text)
  combined      : weighted mix of both, so a very relevant article from a
                  slightly less similar ticket can still rank near the top.
Article content is returned in full; nothing is summarised or generated.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import CLEANED_ARTICLES, Config
from embedding_service import EmbeddingService
from ticket_preprocessor import build_article_text

log = logging.getLogger(__name__)

ARTICLE_FIELDS = ["Id", "Title_article", "ArticleTypeId", "BodyText_article", "TicketNumber", "CreateDate"]


class ArticleRetriever:
    def __init__(self, cfg: Config, embedder: EmbeddingService | None = None) -> None:
        self.path = cfg.artifact(CLEANED_ARTICLES)
        self.rank_cfg = cfg["article_ranking"]
        self.embedder = embedder

    @property
    def available(self) -> bool:
        return self.path.exists()

    def fetch(self, ticket_numbers: list[str]) -> pd.DataFrame:
        if not ticket_numbers or not self.available:
            return pd.DataFrame(columns=ARTICLE_FIELDS)
        table = pq.read_table(self.path, columns=ARTICLE_FIELDS,
                              filters=[("TicketNumber", "in", list(ticket_numbers))])
        return table.to_pandas().drop_duplicates(subset=["Id"], keep="last")

    def get_relevant_articles(self, similar_tickets: list[dict], query_vector: np.ndarray | None = None):
        """Return (articles_by_ticket, ranking, warnings).

        articles_by_ticket: TicketNumber -> list of full articles, best first
        ranking:            every article (without body) ordered by relevance
        """
        warnings: list[str] = []
        if not self.available:
            warnings.append("No article index available yet - run: python build_index.py --stage articles")
            return {}, [], warnings

        source_similarity = {t["TicketNumber"]: float(t["similarity_score"]) for t in similar_tickets}
        articles = self.fetch(list(source_similarity))
        if articles.empty:
            if similar_tickets:
                warnings.append("No linked resolution article was found for the similar tickets.")
            return {}, [], warnings

        limit = int(self.rank_cfg.get("max_articles_per_ticket", 25))
        counts = articles.groupby("TicketNumber")["Id"].transform("size")
        if (counts > limit).any():
            warnings.append(f"Some tickets have more than {limit} articles; only the latest {limit} are returned.")
            articles = articles.sort_values("CreateDate", na_position="first").groupby("TicketNumber").tail(limit)

        articles = articles.reset_index(drop=True)
        articles["source_ticket_similarity"] = articles["TicketNumber"].map(source_similarity).astype(float)
        articles["article_similarity"] = self._article_similarity(articles, query_vector)
        articles["relevance_score"] = self._relevance(articles)
        articles = articles.sort_values(["relevance_score", "Id"], ascending=[False, True], kind="stable")

        records = articles.replace({np.nan: None}).to_dict(orient="records")
        for record in records:
            for key in ("source_ticket_similarity", "article_similarity", "relevance_score"):
                if record[key] is not None:
                    record[key] = round(float(record[key]), 4)

        by_ticket: dict[str, list[dict]] = {}
        for record in records:
            by_ticket.setdefault(record["TicketNumber"], []).append(record)
        ranking = [{k: v for k, v in r.items() if k != "BodyText_article"} for r in records]
        return by_ticket, ranking, warnings

    def _article_similarity(self, articles: pd.DataFrame, query_vector) -> pd.Series:
        if self.rank_cfg.get("mode", "combined") == "source_ticket" or query_vector is None or self.embedder is None:
            return pd.Series(np.nan, index=articles.index)
        max_chars = int(self.rank_cfg.get("max_chars_for_embedding", 2000))
        texts = [build_article_text(t, b, max_chars) for t, b in zip(articles["Title_article"], articles["BodyText_article"])]
        vectors = self.embedder.encode(texts)
        return pd.Series(vectors @ query_vector, index=articles.index)

    def _relevance(self, articles: pd.DataFrame) -> pd.Series:
        mode = self.rank_cfg.get("mode", "combined")
        source, article = articles["source_ticket_similarity"], articles["article_similarity"]
        if mode == "source_ticket" or article.isna().all():
            return source
        if mode == "article_text":
            return article
        w_source = float(self.rank_cfg.get("source_ticket_weight", 0.5))
        w_article = float(self.rank_cfg.get("article_text_weight", 0.5))
        return (w_source * source + w_article * article) / (w_source + w_article)
