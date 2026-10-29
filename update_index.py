"""Incrementally updates the index with new, changed and removed tickets.

    python update_index.py                         # re-read the configured ticket CSVs
    python update_index.py --tickets new_part.csv  # read only these files (a delta)
    python update_index.py --snapshot              # input is a full export: tickets missing from it are removed
    python update_index.py --full                  # force a complete rebuild

What happens per ticket in the input:
  new TicketNumber               -> embedded and appended
  ticket_text changed            -> old vector switched off, new one embedded and appended
  only metadata changed          -> metadata updated in place, no re-embedding
  no longer eligible (reopened)  -> switched off
  missing from a --snapshot      -> switched off

A full rebuild runs automatically when the embedding model, the text
preparation or the storage format differ from what the index was built with.
Articles are refreshed afterwards when an article CSV is configured.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from build_index import STAGES, load_cleaned_tickets, make_manifest, run_articles, run_stages
from config import (CLEANED_TICKETS, EMBEDDINGS_DIR, FAISS_INDEX, INDEX_MANIFEST, TICKET_METADATA, Config,
                    load_config, setup_logging)
from data_cleaner import clean_tickets
from embedding_service import EmbeddingService
from storage_utils import AtomicBatch, load_json, write_json, write_parquet
from ticket_preprocessor import preprocessing_fingerprint
from vector_store import EmbeddingStore, build_metadata, read_faiss_index, write_faiss_index

log = logging.getLogger("update_index")


def full_rebuild_reason(cfg: Config, manifest: dict | None) -> str | None:
    if not manifest or not cfg.artifact(TICKET_METADATA).exists():
        return "no existing index"
    if manifest["model"]["model_name"] != cfg["embedding"]["model_name"]:
        return f"embedding model changed ({manifest['model']['model_name']} -> {cfg['embedding']['model_name']})"
    if manifest["preprocessing"]["fingerprint"] != preprocessing_fingerprint(cfg["text"]):
        return "text preparation settings changed"
    if manifest["embedding_storage"]["dtype"] != cfg["embedding"]["storage_dtype"]:
        return "embedding storage dtype changed"
    return None


def run_update(cfg: Config, ticket_files: list[Path] | None = None, snapshot: bool = False,
               force_full: bool = False) -> bool:
    """Returns True when a full rebuild was run instead (articles are then already rebuilt)."""
    manifest = load_json(cfg.artifact(INDEX_MANIFEST))
    reason = "requested with --full" if force_full else full_rebuild_reason(cfg, manifest)
    if reason:
        log.warning("Running a full rebuild: %s", reason)
        run_stages(cfg, STAGES)
        return True

    incoming_result = clean_tickets(cfg, ticket_files)
    incoming = incoming_result.data
    metadata = pd.read_parquet(cfg.artifact(TICKET_METADATA))
    cleaned = load_cleaned_tickets(cfg)

    active = metadata[metadata["is_active"]]
    current_hash = dict(zip(active["TicketNumber"], active["text_hash"]))
    current_row = dict(zip(active["TicketNumber"], active["vector_id"]))

    known = incoming["TicketNumber"].isin(current_hash.keys())
    text_changed = known & (incoming["TicketNumber"].map(current_hash) != incoming["text_hash"])
    to_embed = incoming[~known | text_changed]

    # Same text: keep only tickets whose metadata really differs from the stored row.
    meta_columns = [c for c in metadata.columns if c in incoming.columns and c not in ("TicketNumber", "text_hash")]
    same_text = incoming[known & ~text_changed]
    rows = same_text["TicketNumber"].map(current_row).to_numpy()
    stored = metadata.loc[rows, meta_columns].fillna("").astype(str).to_numpy()
    differs = (stored != same_text[meta_columns].fillna("").astype(str).to_numpy()).any(axis=1)
    metadata_only, metadata_rows = same_text[differs], rows[differs]

    incoming_ids = set(incoming["TicketNumber"])
    no_longer_eligible = (incoming_result.seen_ids - incoming_ids) & current_hash.keys()
    removed = (current_hash.keys() - incoming_result.seen_ids) if snapshot else set()
    switch_off = set(incoming.loc[text_changed, "TicketNumber"]) | no_longer_eligible | removed

    counts = {"new": int((~known).sum()), "text_changed": int(text_changed.sum()),
              "metadata_refreshed": len(metadata_only), "no_longer_eligible": len(no_longer_eligible),
              "removed_from_snapshot": len(removed)}
    log.info("Changes detected: %s", counts, extra={"data": counts})
    if not len(to_embed) and not switch_off and metadata_only.empty:
        log.info("Index already up to date")
        return False

    # Metadata-only changes are written into the existing rows.
    metadata.loc[metadata_rows, meta_columns] = metadata_only[meta_columns].to_numpy()
    metadata.loc[metadata["TicketNumber"].isin(switch_off), "is_active"] = False

    store = EmbeddingStore(cfg.artifact(EMBEDDINGS_DIR), cfg["embedding"]["storage_dtype"]).open()
    index = read_faiss_index(cfg.artifact(FAISS_INDEX), cfg["index"])
    if not (len(metadata) == store.n_rows == index.ntotal):
        raise RuntimeError("Existing artifacts are out of sync; run a full rebuild: python update_index.py --full")

    with AtomicBatch() as batch:
        if len(to_embed):
            log.info("Embedding %s new or changed tickets", f"{len(to_embed):,}")
            vectors = EmbeddingService(cfg["embedding"]).encode(to_embed["ticket_text"].tolist(), show_progress=True)
            store.append_shard(vectors, batch)
            index.add(vectors)
            appended = build_metadata(to_embed, first_vector_id=len(metadata)).reindex(columns=metadata.columns)
            metadata = pd.concat([metadata, appended], ignore_index=True)

        cleaned = cleaned[~cleaned["TicketNumber"].isin(incoming_ids | switch_off)]
        cleaned = pd.concat([cleaned, incoming], ignore_index=True)
        cleaned = cleaned.sort_values("TicketNumber", kind="stable").reset_index(drop=True)

        trained_on = manifest["faiss"].get("trained_on_rows", index.ntotal)
        if manifest["faiss"]["spec"] != "Flat" and index.ntotal > 1.3 * trained_on:
            log.warning("IVF index has grown >30%% since training; schedule a full rebuild for best recall")

        new_manifest = make_manifest(cfg, store, metadata, {**manifest["faiss"], "ntotal": int(index.ntotal)},
                                     manifest, "incremental_update", counts)
        write_faiss_index(index, batch.stage(cfg.artifact(FAISS_INDEX)))
        write_parquet(metadata, batch.stage(cfg.artifact(TICKET_METADATA)))
        write_parquet(cleaned, batch.stage(cfg.artifact(CLEANED_TICKETS)))
        write_json(new_manifest, batch.stage(cfg.artifact(INDEX_MANIFEST)))
        write_json(incoming_result.report,
                   batch.stage(cfg.resolve(cfg["paths"]["log_dir"]) / f"update_{new_manifest['dataset_version']}.json"))
    log.info("Index version %s: %s vectors, %s active tickets", new_manifest["index_version"],
             f"{len(metadata):,}", f"{int(metadata['is_active'].sum()):,}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Incrementally update the Go-Care recommendation index.")
    parser.add_argument("--tickets", nargs="+", type=Path, help="Ticket CSV files to read (default: config paths)")
    parser.add_argument("--snapshot", action="store_true", help="Input is a full export; remove missing tickets")
    parser.add_argument("--full", action="store_true", help="Force a complete rebuild")
    parser.add_argument("--skip-articles", action="store_true", help="Do not refresh the article store")
    parser.add_argument("--config", help="Path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)
    try:
        files = [p.resolve() for p in args.tickets] if args.tickets else None
        rebuilt = run_update(cfg, files, args.snapshot, args.full)
        if not rebuilt and not args.skip_articles:
            run_articles(cfg)
    except Exception:
        log.exception("Update failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
