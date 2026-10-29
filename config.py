"""Settings loader, artifact file names and logging setup."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.yaml"

# Downloaded models (bge-m3, ~2.3 GB) are kept inside the project, next to .venv.
# Set HF_HOME yourself to use a shared cache instead.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

# Artifact file names (all inside paths.artifacts_dir).
CLEANED_TICKETS = "cleaned_tickets.parquet"
CLEANED_ARTICLES = "cleaned_articles.parquet"
TICKET_METADATA = "ticket_metadata.parquet"
EMBEDDINGS_DIR = "embeddings"
FAISS_INDEX = "ticket_index.faiss"
INDEX_MANIFEST = "index_manifest.json"
CLEANING_REPORT = "cleaning_report.json"
ARTICLE_REPORT = "article_report.json"


class Config(dict):
    """The parsed config.yaml with helpers for resolving paths."""

    @staticmethod
    def resolve(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self["paths"]["artifacts_dir"])

    def artifact(self, name: str) -> Path:
        return self.artifacts_dir / name

    @property
    def null_tokens(self) -> set[str]:
        return set(self["csv"]["null_tokens"])


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("GOCARE_CONFIG") or DEFAULT_CONFIG_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        cfg = Config(yaml.safe_load(handle) or {})
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return cfg


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Pass structured fields with extra={"data": {...}}."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        data = getattr(record, "data", None)
        if data:
            payload["data"] = data
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(cfg: Config) -> None:
    """Readable logs on the console, structured JSON logs in the log directory."""
    log_cfg = cfg.get("logging", {})
    log_dir = cfg.resolve(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(log_cfg.get("level", "INFO"))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / log_cfg.get("file_name", "gocare.log"),
        maxBytes=int(log_cfg.get("rotate_max_bytes", 20_971_520)),
        backupCount=int(log_cfg.get("rotate_backups", 5)),
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    for noisy in ("urllib3", "huggingface_hub", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
