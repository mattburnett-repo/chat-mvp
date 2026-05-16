"""Shared Hugging Face embedding logic (Inference API).

Used by corpus/embed.py (batch ingest), backend/main.py (query-time search),
and corpus/test_ingest.py. Query prefix lives in backend/prompts.py.
"""

import math
import os
from functools import lru_cache
from typing import Any

from huggingface_hub import InferenceClient

from backend.prompts import BGE_QUERY_PREFIX


def embedding_model_name() -> str:
    model = os.environ.get("EMBEDDING_MODEL", "").strip()
    if not model:
        raise ValueError("Missing EMBEDDING_MODEL")
    return model


def _hf_token() -> str:
    token = os.environ.get("PRIMARY_LLM_KEY", "").strip()
    if not token:
        raise ValueError("Missing PRIMARY_LLM_KEY (required for Hugging Face Inference API)")
    return token


@lru_cache(maxsize=1)
def _client() -> InferenceClient:
    return InferenceClient(token=_hf_token())


def _mean_pool(features: Any) -> list[float]:
    """Collapse token-level feature rows to one L2-normalized vector."""
    if hasattr(features, "tolist"):
        features = features.tolist()
    if not features:
        return []
    if isinstance(features[0], (int, float)):
        vec = [float(x) for x in features]
    else:
        rows = [[float(x) for x in row] for row in features]
        dim = len(rows[0])
        sums = [0.0] * dim
        for row in rows:
            for i, val in enumerate(row):
                sums[i] += val
        n = float(len(rows))
        vec = [s / n for s in sums]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _embed_text(text: str) -> list[float]:
    raw = _client().feature_extraction(text, model=embedding_model_name())
    return _mean_pool(raw)


def embed_document(text: str) -> list[float]:
    return _embed_text(text)


def embed_query(text: str) -> list[float]:
    return _embed_text(f"{BGE_QUERY_PREFIX}{text}")
