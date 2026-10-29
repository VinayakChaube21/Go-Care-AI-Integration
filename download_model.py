"""Download the embedding model and check that it works, before starting the long build.

    python download_model.py

The model (BAAI/bge-m3, ~2.3 GB) is stored in <project>/.hf_cache (or $HF_HOME)
and reused by build_index.py, main.py and app.py. Running this is optional:
build_index.py downloads the model itself on first use.
"""
from __future__ import annotations

import sys
import time

from config import load_config, setup_logging
from embedding_service import PROBE_TEXT, EmbeddingModelError, EmbeddingService


def main() -> int:
    cfg = load_config()
    setup_logging(cfg)
    service = EmbeddingService(cfg["embedding"])
    try:
        service.model  # downloads on first use, then loads from the cache
    except EmbeddingModelError as exc:
        print(f"\nFAILED: {exc}\nSee README > 'Model download' for servers without internet access.")
        return 1

    started = time.time()
    service.encode([PROBE_TEXT] * 32)
    rate = 32 / (time.time() - started)
    print(f"\nOK: {service.model_name} | dimension {service.dimension} | device {service.device} | "
          f"~{rate:.1f} texts/s on a short test text (real tickets are longer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
