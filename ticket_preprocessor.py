"""Text normalisation and the canonical `ticket_text` that is embedded.

Historical tickets (during the build) and incoming tickets (at query time) go
through exactly the same functions in this module, so their embeddings are
comparable. Two levels of cleaning exist:

* clean_markup()          - for display: fixes encoding, removes HTML/markup,
                            control characters and repeated whitespace.
* prepare_for_embedding() - additionally removes boilerplate, URLs and
                            personal data that carry no meaning for similarity.

Technical content (error codes, progress codes, product names, Dutch text)
is kept as-is.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata

try:  # optional: repairs mojibake such as "Ã©" -> "é"
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None

# Bump when the cleaning logic below changes; a different version forces a full rebuild.
PREPROCESSING_VERSION = "1.0"

NULL_TEXT = {"", "null", "nan", "none", "n/a"}

RE_ESCAPED_CONTROL = re.compile(r"\\u00[01][0-9a-fA-F]")    # literal "\u0000" from exports
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏﻿�]")
RE_MOJIBAKE = re.compile(r"Ã.|Â|â€")
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_BLOCK_TAG = re.compile(r"<\s*(?:br|/p|/div|/li|/tr|/h[1-6])\s*/?\s*>", re.I)
RE_HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]{0,300}>")
RE_MD_EMPHASIS = re.compile(r"\*{2,}|_{2,}")
RE_SPACES = re.compile(r"[ \t ]+")
RE_BLANK_LINES = re.compile(r"\n{3,}")

RE_IMPORTED_PREFIX = re.compile(
    r"imported ticket\s+of type\s+\d+\.\s*gesprekslog and state \w+\.\s*(?:questionaire\s*:\s*)?", re.I)
RE_PATH_SEPARATOR = re.compile(r"\s*::\s*")                  # "Levering::Installatie"
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
RE_PHONE = re.compile(r"(?<!\w)(?:(?:\+|00)31[\s-]?(?:\(0\)[\s-]?)?\d(?:[\s-]?\d){8}|0\d(?:[\s-]?\d){8})(?!\w)")


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return str(value).strip().lower() in NULL_TEXT


def clean_markup(value) -> str:
    """Display-level cleaning. Keeps all content, removes only markup and noise characters."""
    if is_missing(value):
        return ""
    text = str(value)
    if ftfy is not None and RE_MOJIBAKE.search(text):
        text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = RE_ESCAPED_CONTROL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_CONTROL.sub(" ", text)
    text = RE_HTML_COMMENT.sub(" ", text)
    text = RE_BLOCK_TAG.sub("\n", text)
    text = RE_HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    text = RE_MD_EMPHASIS.sub("", text)
    text = RE_SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def mask_personal_data(text: str, emails: bool = True, phones: bool = True) -> str:
    if emails:
        text = RE_EMAIL.sub("[email]", text)
    if phones:
        text = RE_PHONE.sub("[telefoon]", text)
    return text


def prepare_for_embedding(text: str, text_cfg: dict, max_chars: int) -> str:
    """Remove what carries no meaning for similarity, then truncate on a word boundary."""
    text = clean_markup(text)
    if text_cfg.get("strip_boilerplate", True):
        text = RE_IMPORTED_PREFIX.sub("", text)
        text = RE_PATH_SEPARATOR.sub(" > ", text)
    if text_cfg.get("remove_urls", True):
        text = RE_URL.sub(" ", text)
    text = mask_personal_data(text, text_cfg.get("mask_emails", True), text_cfg.get("mask_phone_numbers", True))
    text = RE_SPACES.sub(" ", text)
    text = RE_BLANK_LINES.sub("\n\n", "\n".join(line.strip() for line in text.split("\n"))).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


def build_ticket_text(title, description, text_cfg: dict, additional_information=None) -> str:
    """The canonical text that is embedded:

        Title: {title}
        Description: {description}

    Ticket IDs and other administrative metadata are deliberately not included.
    """
    title = prepare_for_embedding(title, text_cfg, int(text_cfg.get("max_title_chars", 300)))
    description = prepare_for_embedding(description, text_cfg, int(text_cfg.get("max_body_chars", 4000)))
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")
    if text_cfg.get("include_additional_information") and not is_missing(additional_information):
        extra = prepare_for_embedding(additional_information, text_cfg, 1000)
        if extra:
            parts.append(f"Additional information: {extra}")
    return "\n".join(parts)


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def preprocessing_fingerprint(text_cfg: dict) -> str:
    """Identifies the exact text preparation; stored in the manifest."""
    payload = json.dumps({"version": PREPROCESSING_VERSION, "text": text_cfg}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
