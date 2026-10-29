"""Builds everything with one command:

    python build_index.py                     # all stages
    python build_index.py --stage clean       # only cleaning (no embedding)
    python build_index.py --stage articles    # add/refresh articles later

Stages (always run in this order):
  clean     raw ticket CSV -> cleaned_tickets.parquet + cleaning_report.json
  embed     ticket_text    -> embeddings/part-*.npy  (resumable: restart skips finished shards)
  index     embeddings     -> ticket_index.faiss + ticket_metadata.parquet + index_manifest.json
  articles  raw article CSV -> cleaned_articles.parquet + article_report.json
            (skipped with a warning while no article CSV exists)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq

from config import (ARTICLE_REPORT, CLEANED_ARTICLES, CLEANED_TICKETS, CLEANING_REPORT, EMBEDDINGS_DIR,
                    FAISS_INDEX, INDEX_MANIFEST, TICKET_METADATA, Config, load_config, setup_logging)
from data_cleaner import clean_articles, clean_tickets
from data_loader import resolve_files
from embedding_service import EmbeddingService, embed_to_store
from storage_utils import AtomicBatch, load_json, save_json, save_parquet, write_json, write_parquet
from ticket_preprocessor import PREPROCESSING_VERSION, preprocessing_fingerprint
from vector_store import (TEXT_COLUMNS, EmbeddingStore, build_faiss_index, build_metadata, compute_dataset_hash,
                          write_faiss_index)

log = logging.getLogger("build_index")

STAGES = ["clean", "embed", "index", "articles"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def load_cleaned_tickets(cfg: Config, columns: list[str] | None = None, with_text: bool = True) -> pd.DataFrame:
    path = cfg.artifact(CLEANED_TICKETS)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python build_index.py --stage clean")
    if not with_text:
        columns = [c for c in pq.read_schema(path).names if c not in TEXT_COLUMNS]
    return pd.read_parquet(path, columns=columns)


def source_file_info(cfg: Config, kind: str) -> list[dict]:
    return [{"file": f.name, "size_bytes": f.stat().st_size,
             "modified": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat()}
            for f in resolve_files(cfg["paths"][f"{kind}_csv"])]


def make_manifest(cfg: Config, store: EmbeddingStore, metadata: pd.DataFrame, faiss_info: dict,
                  previous: dict | None, event: str, details: dict | None = None) -> dict:
    """index_manifest.json: what was built, from what, with which model and settings."""
    now = _now()
    previous = previous or {}
    version = int(previous.get("index_version", 0)) + 1
    history = list(previous.get("history", []))[-49:]
    history.append({"index_version": version, "at": now.isoformat(), "event": event, **(details or {})})
    return {
        "index_version": version,
        "dataset_version": now.strftime("%Y%m%dT%H%M%SZ"),
        "created_at": previous.get("created_at", now.isoformat()) if event != "full_build" else now.isoformat(),
        "updated_at": now.isoformat(),
        "dataset_hash": store.state.get("dataset_hash"),
        "rows": {"vectors": int(len(metadata)), "active_tickets": int(metadata["is_active"].sum())},
        "model": store.state["model_info"],
        "probe_embedding": store.state["probe_embedding"],
        "preprocessing": {"version": PREPROCESSING_VERSION, "fingerprint": preprocessing_fingerprint(cfg["text"])},
        "embedding_storage": {"dtype": store.dtype.name, "shards": len(store.state["shards"])},
        "faiss": {**faiss_info, "nprobe": cfg["index"].get("ivf_nprobe")},
        "source_files": source_file_info(cfg, "tickets"),
        "files": {"cleaned_tickets": CLEANED_TICKETS, "ticket_metadata": TICKET_METADATA,
                  "embeddings_dir": EMBEDDINGS_DIR, "faiss_index": FAISS_INDEX, "articles": CLEANED_ARTICLES},
        "history": history,
    }


# -----------------------------------------------------------------------------
# Stages
# -----------------------------------------------------------------------------
def run_clean(cfg: Config) -> None:
    result = clean_tickets(cfg)
    save_json(result.report, cfg.artifact(CLEANING_REPORT))
    if result.data.empty:
        raise RuntimeError(f"No tickets left after cleaning - see {cfg.artifact(CLEANING_REPORT)}")
    save_parquet(result.data, cfg.artifact(CLEANED_TICKETS))
    log.info("Saved %s cleaned tickets -> %s", f"{len(result.data):,}", CLEANED_TICKETS)


def run_embed(cfg: Config) -> None:
    emb_cfg = cfg["embedding"]
    tickets = pd.read_parquet(cfg.artifact(CLEANED_TICKETS), columns=["TicketNumber", "ticket_text", "text_hash"]) \
        if cfg.artifact(CLEANED_TICKETS).exists() else load_cleaned_tickets(cfg)
    store = EmbeddingStore(cfg.artifact(EMBEDDINGS_DIR), emb_cfg["storage_dtype"])
    store.prepare_for(compute_dataset_hash(tickets), emb_cfg["model_name"])

    service = EmbeddingService(emb_cfg)
    log.info("Embedding %s tickets with %s", f"{len(tickets):,}", emb_cfg["model_name"])
    embed_to_store(service, tickets["ticket_text"], store, int(emb_cfg.get("shard_rows", 50_000)))
    store.state["model_info"] = service.info()
    store.state["probe_embedding"] = service.probe_vector()
    store.save_state()


def run_index(cfg: Config) -> None:
    tickets = load_cleaned_tickets(cfg, with_text=False)
    store = EmbeddingStore(cfg.artifact(EMBEDDINGS_DIR), cfg["embedding"]["storage_dtype"])
    if store.state.get("dataset_hash") != compute_dataset_hash(tickets) or store.n_rows != len(tickets) \
            or "model_info" not in store.state:
        raise RuntimeError("Embeddings do not match cleaned_tickets.parquet. Run: python build_index.py --stage embed")
    store.open()

    index, faiss_info = build_faiss_index(store, cfg["index"])
    metadata = build_metadata(tickets)
    previous = load_json(cfg.artifact(INDEX_MANIFEST))
    manifest = make_manifest(cfg, store, metadata, {**faiss_info, "trained_on_rows": len(metadata)},
                             previous, "full_build", {"vectors": len(metadata)})
    with AtomicBatch() as batch:
        write_faiss_index(index, batch.stage(cfg.artifact(FAISS_INDEX)))
        write_parquet(metadata, batch.stage(cfg.artifact(TICKET_METADATA)))
        write_json(manifest, batch.stage(cfg.artifact(INDEX_MANIFEST)))
    log.info("Index version %s written: %s vectors", manifest["index_version"], f"{len(metadata):,}")


def run_articles(cfg: Config) -> None:
    files = resolve_files(cfg["paths"]["articles_csv"])
    if not files:
        log.warning("No article CSV matches %s - article stage skipped. Add the file and run: "
                    "python build_index.py --stage articles", cfg["paths"]["articles_csv"])
        return
    ticket_ids = None
    if cfg["articles"].get("restrict_to_indexed_tickets", True):
        ticket_ids = set(pd.read_parquet(cfg.artifact(CLEANED_TICKETS), columns=["TicketNumber"])["TicketNumber"]) \
            if cfg.artifact(CLEANED_TICKETS).exists() else None
        if ticket_ids is None:
            log.warning("cleaned_tickets.parquet not found - articles are not restricted to indexed tickets")

    result = clean_articles(cfg, ticket_ids, files)
    path = cfg.artifact(CLEANED_ARTICLES)
    if path.exists():
        result.report["changes_since_previous_build"] = _article_changes(pd.read_parquet(path, columns=["Id", "content_hash"]),
                                                                       result.data)
    save_parquet(result.data, path, row_group_size=20_000)
    save_json(result.report, cfg.artifact(ARTICLE_REPORT))
    log.info("Saved %s articles for %s tickets -> %s", f"{len(result.data):,}",
             f"{result.report['tickets_with_articles']:,}", CLEANED_ARTICLES)


def _article_changes(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    old_hash = dict(zip(old["Id"], old["content_hash"]))
    new_hash = dict(zip(new["Id"], new["content_hash"]))
    return {
        "new": sum(1 for i in new_hash if i not in old_hash),
        "updated": sum(1 for i, h in new_hash.items() if i in old_hash and old_hash[i] != h),
        "removed": sum(1 for i in old_hash if i not in new_hash),
    }


STAGE_FUNCTIONS = {"clean": run_clean, "embed": run_embed, "index": run_index, "articles": run_articles}


def run_stages(cfg: Config, stages: list[str]) -> None:
    for stage in [s for s in STAGES if s in stages]:
        started = time.time()
        log.info("=== Stage %s ===", stage)
        STAGE_FUNCTIONS[stage](cfg)
        log.info("=== Stage %s finished in %.1f min ===", stage, (time.time() - started) / 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Go-Care ticket recommendation index.")
    parser.add_argument("--stage", action="append", choices=STAGES,
                        help="Run only this stage (repeatable). Default: all stages.")
    parser.add_argument("--config", help="Path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)
    try:
        run_stages(cfg, args.stage or STAGES)
    except Exception:
        log.exception("Build failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
