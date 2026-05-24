"""Evaluation harness: reference cases + live RAG runs via backend.rag."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from backend.rag import RAGQueryResult, run_rag_query

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
REFERENCE_CASES_PATH = LIB_DIR / "reference_cases.json"

load_dotenv(PROJECT_ROOT / ".env")


def load_reference_cases(path: Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads((path or REFERENCE_CASES_PATH).read_text(encoding="utf-8"))
    return list(payload["cases"])


def case_result_from_rag(
    case: dict[str, Any], result: RAGQueryResult
) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "question": case["question"],
        "ground_truth": case.get("ground_truth", ""),
        "generated_answer": result.answer,
        "retrieved_contexts": result.retrieved_contexts,
        "sources": [
            {"source_url": s.source_url, "chunk_index": s.chunk_index}
            for s in result.sources
        ],
        "latency_ms": result.latency_ms,
        "expected_source_urls": case.get("expected_source_urls", []),
        "expected_aspects": case.get("expected_aspects", []),
        "expect_no_match": case.get("expect_no_match", False),
    }


def run_case(case: dict[str, Any], top_k: int | None = None) -> dict[str, Any]:
    result = run_rag_query(case["question"], top_k=top_k)
    return case_result_from_rag(case, result)


def run_all_cases(
    cases: list[dict[str, Any]] | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    items = cases if cases is not None else load_reference_cases()
    return [run_case(case, top_k=top_k) for case in items]


def to_ragas_test_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in results:
        contexts = row["retrieved_contexts"]
        if not contexts:
            contexts = [""]
        out.append(
            {
                "question": row["question"],
                "generated_answer": row["generated_answer"],
                "retrieved_contexts": contexts,
                "ground_truth": row["ground_truth"],
            }
        )
    return out
