"""Split page text into overlapping chunks for embedding."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from utils.env_loader import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

# ~500 tokens at ~4 chars/token; overlap preserves context across boundaries.
CHUNK_MAX_CHARS = int(os.environ["CHUNK_MAX_CHARS"].strip())
CHUNK_OVERLAP_CHARS = int(os.environ["CHUNK_OVERLAP_CHARS"].strip())


def chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Return non-empty chunks; empty input yields an empty list."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            break_at = text.rfind(" ", start + max(1, max_chars // 2), end)
            if break_at > start:
                end = break_at + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, max_chars // 2)
        start = next_start

    return chunks if chunks else [text[:max_chars].strip()]
