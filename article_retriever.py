"""Fetches all resolution articles linked to the similar tickets.

Articles are NOT embedded. They live in cleaned_articles.parquet, sorted by
TicketNumber; a lookup filters on TicketNumber, so pyarrow reads only the
row groups that can contain those tickets. The sorted file works as the
ticket -> article index.

Per ticket, articles are returned in the order they were written (the
resolution sequence). The overall ranking follows the similarity of the
ticket each article belongs to. Article content is returned in full;
nothing is summarised or generated.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import CLEANED_ARTICLES, Config

log = logging.getLogger(__name__)

ARTICLE_FIELDS = ["Id", "Title_article", "ArticleTypeId", "BodyText_article", "TicketNumber", "CreateDate"]


class ArticleRetriever:
    def __init__(self, cfg: Config) -> None:
        self.path = cfg.artifact(CLEANED_ARTICLES)
        self.max_per_ticket = int((cfg.get("article_retrieval") or {}).get("max_articles_per_ticket", 0))

    @property
    def available(self) -> bool:
        return self.path.exists()

    def fetch(self, ticket_numbers: list[str]) -> pd.DataFrame:
        if not ticket_numbers or not self.available:
            return pd.DataFrame(columns=ARTICLE_FIELDS)
        table = pq.read_table(self.path, columns=ARTICLE_FIELDS,
                              filters=[("TicketNumber", "in", list(ticket_numbers))])
        return table.to_pandas().drop_duplicates(subset=["Id"], keep="last")

    def get_relevant_articles(self, similar_tickets: list[dict]):
        """Return (articles_by_ticket, ranking, warnings).

        articles_by_ticket: TicketNumber -> list of full articles, in the order they were written
        ranking:            every article (without body), most similar source ticket first
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

        articles = articles.sort_values(["TicketNumber", "CreateDate", "Id"], na_position="first", kind="stable")
        if self.max_per_ticket > 0:
            counts = articles.groupby("TicketNumber")["Id"].transform("size")
            if (counts > self.max_per_ticket).any():
                warnings.append(f"Some tickets have more than {self.max_per_ticket} articles; "
                                f"only the latest {self.max_per_ticket} are returned.")
                articles = articles.groupby("TicketNumber").tail(self.max_per_ticket)

        articles["source_ticket_similarity"] = articles["TicketNumber"].map(source_similarity).round(4)
        records = articles.replace({np.nan: None}).to_dict(orient="records")

        by_ticket: dict[str, list[dict]] = {}
        for record in records:
            by_ticket.setdefault(record["TicketNumber"], []).append(record)
        ranking = sorted(({k: v for k, v in r.items() if k != "BodyText_article"} for r in records),
                         key=lambda r: -r["source_ticket_similarity"])
        return by_ticket, ranking, warnings
