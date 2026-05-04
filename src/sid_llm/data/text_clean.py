"""Lightweight text cleaning for product titles/descriptions.

Strips HTML, normalizes whitespace, deduplicates repeated keyword tokens
(case-insensitive global dedup; useful for Amazon's keyword-stuffed titles),
truncates. The LLM-based refinement (Walmart's Summarizer->Evaluator->Refiner)
is deferred to v1.
"""
from __future__ import annotations

import re

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def clean_text(s: str | None, max_chars: int = 500) -> str:
    if not s:
        return ""
    # 1. strip HTML
    s = _HTML_TAG.sub(" ", s)
    # 2. normalize whitespace
    s = _WHITESPACE.sub(" ", s).strip()
    # 3. dedup repeated tokens, case-insensitive (preserves first-seen casing)
    tokens = s.split()
    deduped: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tok)
    s = " ".join(deduped)
    # 4. truncate
    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s
