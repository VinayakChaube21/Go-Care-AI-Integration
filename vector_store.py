"""Persistent storage of ticket embeddings, the FAISS index and row-aligned metadata.

Row alignment is the central rule: row i of the embedding matrix, vector id i
in FAISS and row i of ticket_metadata.parquet (column vector_id == i) all
describe the same ticket. Rows are only ever appended. A ticket that changes
or disappears is switched off (is_active = False) instead of being removed,
so no vector id ever shifts. A full rebuild compacts everything again.

Embeddings are stored as .npy shards (embeddings/part-00000.npy, ...) listed
in embeddings/shards.json. Shards make the long first build resumable and let
updates append vectors without rewriting gigabytes.
"""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import CLEANED_TICKETS, EMBEDDINGS_DIR, FAISS_INDEX, INDEX_MANIFEST, TICKET_METADATA, Config
from storage_utils import AtomicBatch, load_json, save_json, save_npy, write_json

log = logging.getLogger(__name__)

# Columns of cleaned_tickets that stay out of ticket_metadata (text lives in cleaned_tickets only).
TEXT_COLUMNS = ["Title_ticket", "BodyText_ticket", "AdditionalInformation", "ticket_text", "ResponsibleAgent"]


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss is not installed: pip install faiss-cpu") from exc
    return faiss


def compute_dataset_hash(tickets: pd.DataFrame) -> str:
    """Fingerprint of the exact rows (and their order) that are embedded."""
    digest = hashlib.sha1()
    for number, text_sha in zip(tickets["TicketNumber"], tickets["text_hash"]):
        digest.update(f"{number}:{text_sha}\n".encode())
    return digest.hexdigest()


def build_metadata(tickets: pd.DataFrame, first_vector_id: int = 0) -> pd.DataFrame:
    meta = tickets.drop(columns=[c for c in TEXT_COLUMNS if c in tickets.columns]).copy()
    meta.insert(0, "vector_id", np.arange(first_vector_id, first_vector_id + len(meta), dtype=np.int64))
    meta["is_active"] = True
    return meta.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Embedding shards
# -----------------------------------------------------------------------------
class EmbeddingStore:
    STATE_FILE = "shards.json"

    def __init__(self, directory: Path, storage_dtype: str = "float16") -> None:
        self.dir = directory
        self.dtype = np.dtype(storage_dtype)
        self.state = load_json(directory / self.STATE_FILE, default={"shards": []})
        self._arrays: list[np.ndarray] | None = None
        self._offsets: np.ndarray | None = None

    # ---- writing -------------------------------------------------------------
    def prepare_for(self, dataset_hash: str, model_name: str) -> None:
        """Keep finished shards if they belong to this dataset and model, otherwise start over."""
        same = (self.state.get("dataset_hash") == dataset_hash
                and self.state.get("model_name") == model_name
                and self.state.get("dtype") == self.dtype.name)
        if same:
            return
        if self.state.get("shards"):
            log.info("Dataset or model changed - discarding %d old embedding shards", len(self.state["shards"]))
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in self.dir.glob("part-*.npy"):
            old.unlink()
        self.state = {"dataset_hash": dataset_hash, "model_name": model_name, "dtype": self.dtype.name, "shards": []}
        self.save_state()

    def has_shard(self, number: int, rows: int) -> bool:
        shards = self.state["shards"]
        return number < len(shards) and shards[number]["rows"] == rows and (self.dir / shards[number]["file"]).exists()

    def write_shard(self, number: int, vectors: np.ndarray) -> None:
        if number > len(self.state["shards"]):
            raise ValueError("Shards must be written in order")
        name = f"part-{number:05d}.npy"
        save_npy(vectors.astype(self.dtype), self.dir / name)
        entry = {"file": name, "rows": int(len(vectors))}
        if number == len(self.state["shards"]):
            self.state["shards"].append(entry)
        else:
            self.state["shards"][number] = entry
        self.save_state()

    def append_shard(self, vectors: np.ndarray, batch: AtomicBatch) -> None:
        """Add a shard for an incremental update. The shard list is committed with `batch`."""
        number = len(self.state["shards"])
        name = f"part-{number:05d}.npy"
        save_npy(vectors.astype(self.dtype), self.dir / name)  # not referenced until the batch commits
        self.state["shards"].append({"file": name, "rows": int(len(vectors))})
        self.state["updated_incrementally"] = True
        write_json(self.state, batch.stage(self.dir / self.STATE_FILE))
        self._arrays = None

    def save_state(self) -> None:
        save_json(self.state, self.dir / self.STATE_FILE)

    # ---- reading -------------------------------------------------------------
    def open(self) -> "EmbeddingStore":
        if not self.state.get("shards"):
            raise FileNotFoundError(f"No embeddings in {self.dir}. Run: python build_index.py --stage embed")
        self._arrays = [np.load(self.dir / s["file"], mmap_mode="r") for s in self.state["shards"]]
        sizes = [len(a) for a in self._arrays]
        self._offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        return self

    @property
    def arrays(self) -> list[np.ndarray]:
        if self._arrays is None:
            self.open()
        return self._arrays

    @property
    def n_rows(self) -> int:
        return sum(s["rows"] for s in self.state.get("shards", []))

    @property
    def dim(self) -> int:
        return int(self.arrays[0].shape[1])

    def take(self, rows: np.ndarray) -> np.ndarray:
        """Embeddings of the given row numbers as float32, in the given order."""
        arrays = self.arrays  # opens the shards on first use
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), self.dim), dtype=np.float32)
        shard_of = np.searchsorted(self._offsets, rows, side="right") - 1
        for shard in np.unique(shard_of):
            pick = np.flatnonzero(shard_of == shard)
            out[pick] = arrays[shard][rows[pick] - self._offsets[shard]]
        return out

    def cosine(self, rows: np.ndarray, query: np.ndarray, block_rows: int = 50_000) -> np.ndarray:
        """Exact cosine similarity (dot product of unit vectors) for the given rows."""
        scores = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), block_rows):
            block = rows[start:start + block_rows]
            scores[start:start + len(block)] = self.take(block) @ query
        return scores

    def iter_blocks(self, block_rows: int = 100_000):
        for array in self.arrays:
            for start in range(0, len(array), block_rows):
                yield np.ascontiguousarray(array[start:start + block_rows], dtype=np.float32)


# -----------------------------------------------------------------------------
# FAISS index
# -----------------------------------------------------------------------------
def build_faiss_index(store: EmbeddingStore, index_cfg: dict):
    faiss = _faiss()
    if int(index_cfg.get("faiss_threads", 0)) > 0:
        faiss.omp_set_num_threads(int(index_cfg["faiss_threads"]))

    n, dim = store.n_rows, store.dim
    kind = index_cfg.get("type", "auto")
    if kind == "auto":
        kind = "flat" if n <= int(index_cfg.get("flat_max_rows", 1_500_000)) else "ivf_flat"

    if kind == "flat":
        spec = "Flat"
    else:
        nlist = index_cfg.get("ivf_nlist", "auto")
        nlist = int(4 * math.sqrt(n)) if nlist == "auto" else int(nlist)
        nlist = max(1, min(nlist, 65_536, n // 39))
        spec = f"IVF{nlist},{'SQ8' if kind == 'ivf_sq8' else 'Flat'}"

    log.info("Building FAISS index %s over %s vectors (%d dims)", spec, f"{n:,}", dim)
    index = faiss.index_factory(dim, spec, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        rng = np.random.default_rng(int(index_cfg.get("seed", 42)))
        sample_size = min(n, max(int(index_cfg.get("train_sample_rows", 200_000)), 40 * index.nlist))
        sample = np.sort(rng.choice(n, size=sample_size, replace=False))
        log.info("Training IVF on %s sampled vectors", f"{sample_size:,}")
        index.train(store.take(sample))
    for block in store.iter_blocks():
        index.add(block)
    _set_nprobe(index, index_cfg)
    return index, {"spec": spec, "ntotal": int(index.ntotal), "metric": "inner_product"}


def write_faiss_index(index, path: Path) -> None:
    _faiss().write_index(index, str(path))


def read_faiss_index(path: Path, index_cfg: dict):
    index = _faiss().read_index(str(path))
    _set_nprobe(index, index_cfg)
    return index


def _ivf_of(index):
    return _faiss().try_extract_index_ivf(index)


def _set_nprobe(index, index_cfg: dict) -> None:
    ivf = _ivf_of(index)
    if ivf is not None:
        ivf.nprobe = int(index_cfg.get("ivf_nprobe", 32))


def faiss_search(index, query: np.ndarray, mask: np.ndarray, k: int, nprobe: int) -> np.ndarray:
    """Search only the rows allowed by `mask` (metadata pre-filter inside FAISS)."""
    faiss = _faiss()
    bitmap = np.packbits(mask.astype(np.uint8), bitorder="little")
    selector = faiss.IDSelectorBitmap(len(bitmap), faiss.swig_ptr(bitmap))
    if _ivf_of(index) is not None:
        params = faiss.SearchParametersIVF(sel=selector, nprobe=nprobe)
    else:
        params = faiss.SearchParameters(sel=selector)
    _, ids = index.search(query.reshape(1, -1).astype(np.float32), k, params=params)
    return ids[0][ids[0] >= 0]


# -----------------------------------------------------------------------------
# Query-time view of all stored artifacts
# -----------------------------------------------------------------------------
class VectorStore:
    """Loads the index once at startup and answers similarity searches."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.index_cfg = cfg["index"]
        manifest_path = cfg.artifact(INDEX_MANIFEST)
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found. Build the index first: python build_index.py")
        self.manifest = load_json(manifest_path)
        self.metadata = pd.read_parquet(cfg.artifact(TICKET_METADATA))
        self.embeddings = EmbeddingStore(cfg.artifact(EMBEDDINGS_DIR), cfg["embedding"]["storage_dtype"]).open()
        self.index = read_faiss_index(cfg.artifact(FAISS_INDEX), self.index_cfg)
        self._check_alignment()
        log.info("Vector store loaded: %s vectors, %s active tickets, index version %s",
                 f"{len(self.metadata):,}", f"{int(self.metadata['is_active'].sum()):,}",
                 self.manifest.get("index_version"))

    def _check_alignment(self) -> None:
        sizes = {"metadata": len(self.metadata), "embeddings": self.embeddings.n_rows, "faiss": int(self.index.ntotal)}
        if len(set(sizes.values())) != 1:
            raise RuntimeError(f"Artifacts are out of sync {sizes}. Re-run update_index.py or build_index.py.")
        if not np.array_equal(self.metadata["vector_id"].to_numpy(), np.arange(len(self.metadata))):
            raise RuntimeError("ticket_metadata.vector_id is not 0..n-1; rebuild the index.")

    def search(self, query: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, str]:
        """Top-k rows allowed by `mask`, ranked by exact cosine similarity.

        Few candidates: cosine is computed for every candidate (exact).
        Many candidates: FAISS proposes top_k * overfetch rows from inside the
        mask, then the exact cosine is recomputed from the stored vectors.
        """
        candidates = np.flatnonzero(mask)
        if len(candidates) == 0 or k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), "no_candidates"

        if len(candidates) <= int(self.index_cfg.get("exact_search_max_candidates", 100_000)):
            rows, method = candidates, "exact"
        else:
            fetch = k * int(self.index_cfg.get("faiss_overfetch", 20))
            rows = faiss_search(self.index, query, mask, fetch, int(self.index_cfg.get("ivf_nprobe", 32)))
            rows = np.sort(rows[mask[rows]])  # re-check the filter, never trust the ANN step alone
            method = "faiss_prefilter+exact_rescore"

        scores = self.embeddings.cosine(rows, query)
        order = np.argsort(-scores, kind="stable")[:k]  # ties broken by vector id -> deterministic
        return rows[order], scores[order], method

    def ticket_details(self, ticket_numbers: list[str], columns: list[str] | None = None) -> pd.DataFrame:
        """Full ticket rows (incl. text) from cleaned_tickets.parquet, read via row-group pruning."""
        path = self.cfg.artifact(CLEANED_TICKETS)
        stored = set(pq.read_schema(path).names)
        wanted = columns or sorted(stored)
        if not ticket_numbers:
            return pd.DataFrame(columns=wanted)
        table = pq.read_table(path, columns=[c for c in wanted if c in stored],
                              filters=[("TicketNumber", "in", list(ticket_numbers))])
        # Columns that were not kept during cleaning come back empty instead of failing.
        return table.to_pandas().reindex(columns=wanted)
