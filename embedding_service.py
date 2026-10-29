"""Generates unit-normalised sentence embeddings (default model: BAAI/bge-m3).

The model is loaded once and reused. Because every vector has length 1,
cosine similarity between two tickets is simply their dot product.
"""
from __future__ import annotations

import logging
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from vector_store import EmbeddingStore

log = logging.getLogger(__name__)

# Re-embedded at query start and compared with the vector stored at build time,
# to prove that queries use the same model as the index.
PROBE_TEXT = "Title: Geen internet\nDescription: Klant heeft sinds gisteren geen internetverbinding via glasvezel."


class EmbeddingModelError(RuntimeError):
    pass


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class EmbeddingService:
    def __init__(self, emb_cfg: dict) -> None:
        self.cfg = emb_cfg
        self.model_name = emb_cfg["model_name"]
        self._model = None
        self.device = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError("Install torch and sentence-transformers (see requirements.txt)") from exc

        device = self.cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if device == "cpu" and int(self.cfg.get("cpu_threads", 0)) > 0:
            torch.set_num_threads(int(self.cfg["cpu_threads"]))
        log.info("Loading embedding model %s on %s (%d torch threads)", self.model_name, device,
                 torch.get_num_threads())
        started = time.time()
        try:
            model = SentenceTransformer(self.model_name, device=device, revision=self.cfg.get("model_revision"))
        except Exception as exc:
            raise EmbeddingModelError(f"Could not load embedding model '{self.model_name}': {exc}") from exc
        model.max_seq_length = int(self.cfg.get("max_seq_length", 512))
        if device == "cuda" and self.cfg.get("fp16_on_gpu", True):
            model.half()
        model.eval()
        log.info("Model loaded in %.1fs (dimension=%d)", time.time() - started, self.dimension_of(model))
        return model

    @staticmethod
    def dimension_of(model) -> int:
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())

    @property
    def dimension(self) -> int:
        return self.dimension_of(self.model)

    def encode(self, texts: list[str], batch_size: int | None = None, show_progress: bool = False) -> np.ndarray:
        """Return a float32 matrix (len(texts) x dimension) of unit-length vectors."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=int(batch_size or self.cfg.get("batch_size", 32)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)
        return self._validate(vectors, len(texts))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _validate(self, vectors: np.ndarray, expected_rows: int) -> np.ndarray:
        if vectors.ndim != 2 or vectors.shape != (expected_rows, self.dimension):
            raise EmbeddingModelError(f"Unexpected embedding shape {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise EmbeddingModelError("Embedding contains NaN or infinite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms < 1e-6).any():
            raise EmbeddingModelError("Embedding with zero length produced (empty text?)")
        return vectors / norms  # re-normalise after any fp16 rounding

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_revision": self.cfg.get("model_revision"),
            "dimension": self.dimension,
            "max_seq_length": int(self.model.max_seq_length),
            "device_used_for_build": self.device,
            "sentence_transformers_version": _package_version("sentence-transformers"),
            "torch_version": _package_version("torch"),
        }

    def probe_vector(self) -> list[float]:
        return self.encode_one(PROBE_TEXT).round(6).tolist()

    def check_matches_probe(self, stored_probe: list[float] | None, strict: bool = True) -> float | None:
        """Cosine between the stored probe vector and a fresh one; must be ~1.0."""
        if not stored_probe:
            return None
        similarity = float(np.dot(self.encode_one(PROBE_TEXT), np.asarray(stored_probe, dtype=np.float32)))
        if similarity < 0.99:
            message = (f"Query model does not produce the same vectors as the index "
                       f"(probe cosine {similarity:.4f}). Rebuild the index or fix embedding.model_name.")
            if strict:
                raise EmbeddingModelError(message)
            log.warning(message)
        return similarity


def embed_to_store(service: EmbeddingService, texts: pd.Series, store: EmbeddingStore, shard_rows: int) -> None:
    """Embed `texts` shard by shard. Finished shards are skipped, so a stopped run resumes."""
    total = len(texts)
    n_shards = (total + shard_rows - 1) // shard_rows
    started = time.time()
    embedded_now = 0
    for shard in range(n_shards):
        start, end = shard * shard_rows, min((shard + 1) * shard_rows, total)
        if store.has_shard(shard, end - start):
            log.info("Shard %d/%d already embedded - skipped", shard + 1, n_shards)
            continue
        vectors = service.encode(texts.iloc[start:end].tolist(), show_progress=True)
        store.write_shard(shard, vectors)
        embedded_now += end - start
        rate = embedded_now / max(time.time() - started, 1e-9)
        remaining = sum(min(shard_rows, total - s * shard_rows) for s in range(shard + 1, n_shards))
        log.info("Shard %d/%d done (rows %s-%s) | %.0f tickets/s | ETA %.1f min",
                 shard + 1, n_shards, f"{start:,}", f"{end:,}", rate, remaining / max(rate, 1e-9) / 60)
