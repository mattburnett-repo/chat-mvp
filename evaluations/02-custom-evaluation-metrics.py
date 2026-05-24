"""Custom RAG metrics for the chat MVP reference evaluation set."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from backend.rag import NO_MATCH_ANSWER
from corpus.embeddings import embed_document

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class CustomRAGMetrics:
    """Project-specific metrics over harness output."""

    def _encode_one(self, text: str) -> np.ndarray:
        return np.array(embed_document(text), dtype=float)

    def _encode_many(self, texts: list[str]) -> np.ndarray:
        return np.array([self._encode_one(t) for t in texts])

    def hallucination_score(self, answer: str, contexts: list[str]) -> float:
        """Fraction of answer sentences supported by retrieved context."""
        answer_sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not answer_sentences:
            return 1.0

        combined_context = " ".join(contexts)
        unsupported = sum(
            1
            for sentence in answer_sentences
            if not self._is_supported_by_context(sentence, combined_context)
        )
        return 1.0 - (unsupported / len(answer_sentences))

    def _is_supported_by_context(self, claim: str, context: str) -> bool:
        claim_embedding = self._encode_one(claim)
        context_sentences = [s.strip() for s in context.split(".") if s.strip()]
        if not context_sentences:
            return False

        context_embeddings = self._encode_many(context_sentences)
        similarities = np.dot(context_embeddings, claim_embedding) / (
            np.linalg.norm(context_embeddings, axis=1) * np.linalg.norm(claim_embedding)
        )
        return float(np.max(similarities)) > 0.8

    def completeness_score(
        self, answer: str, question: str, expected_aspects: list[str]
    ) -> float:
        if not expected_aspects:
            return 1.0

        covered = 0
        answer_lower = answer.lower()
        answer_embedding = self._encode_one(answer)

        for aspect in expected_aspects:
            if aspect.lower() in answer_lower:
                covered += 1
                continue
            aspect_embedding = self._encode_one(aspect)
            similarity = np.dot(aspect_embedding, answer_embedding) / (
                np.linalg.norm(aspect_embedding) * np.linalg.norm(answer_embedding)
            )
            if similarity > 0.7:
                covered += 1

        return covered / len(expected_aspects)

    def response_time_score(
        self, response_time_ms: float, target_time_ms: float = 2000
    ) -> float:
        if response_time_ms <= target_time_ms:
            return 1.0
        return float(np.exp(-0.0005 * (response_time_ms - target_time_ms)))

    def source_url_mention_score(self, answer: str, source_urls: list[str]) -> float:
        """Check whether the answer mentions URLs from retrieved sources."""
        if not source_urls:
            return 1.0
        mentioned = sum(1 for url in source_urls if url in answer)
        return mentioned / len(source_urls)

    def expected_source_retrieval_score(
        self,
        sources: list[dict],
        expected_source_urls: list[str],
    ) -> float:
        """Check whether retrieval returned expected corpus URLs."""
        if not expected_source_urls:
            return 1.0
        retrieved = {item["source_url"] for item in sources}
        hits = sum(1 for url in expected_source_urls if url in retrieved)
        return hits / len(expected_source_urls)

    def insufficient_context_score(self, answer: str, expect_no_match: bool) -> float:
        """Score whether no-match / insufficient-context behavior is correct."""
        lowered = answer.lower()
        got_no_match = (
            answer.strip() == NO_MATCH_ANSWER
            or "insufficient" in lowered
            or "no matching" in lowered
        )
        if expect_no_match:
            return 1.0 if got_no_match else 0.0
        return 1.0 if not got_no_match else 0.0


def evaluate_case(metrics: CustomRAGMetrics, row: dict) -> dict[str, float]:
    contexts = row["retrieved_contexts"]
    source_urls = [s["source_url"] for s in row.get("sources", [])]
    return {
        "hallucination_score": metrics.hallucination_score(
            row["generated_answer"], contexts
        ),
        "completeness_score": metrics.completeness_score(
            row["generated_answer"],
            row["question"],
            row.get("expected_aspects", []),
        ),
        "response_time_score": metrics.response_time_score(row["latency_ms"]),
        "source_url_mention_score": metrics.source_url_mention_score(
            row["generated_answer"], source_urls
        ),
        "expected_source_retrieval_score": metrics.expected_source_retrieval_score(
            row.get("sources", []),
            row.get("expected_source_urls", []),
        ),
        "insufficient_context_score": metrics.insufficient_context_score(
            row["generated_answer"],
            row.get("expect_no_match", False),
        ),
    }


def summarize_scores(all_scores: list[dict[str, float]]) -> dict[str, float]:
    if not all_scores:
        return {}
    keys = all_scores[0].keys()
    return {key: float(np.mean([row[key] for row in all_scores])) for key in keys}


def _eval_dir_on_path() -> None:
    root = str(Path(__file__).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)


if __name__ == "__main__":
    _eval_dir_on_path()
    from lib.baseline import check_regression, load_baseline, save_baseline, should_update_baseline
    from lib.harness import run_all_cases

    print("Running reference cases through RAG harness (custom metrics)...")
    metrics = CustomRAGMetrics()
    results = run_all_cases()
    per_case = []
    for row in results:
        scores = evaluate_case(metrics, row)
        per_case.append(scores)
        print(f"\n{row.get('id', row['question'][:40])}:")
        for name, value in scores.items():
            print(f"  {name}: {value:.3f}")

    summary = summarize_scores(per_case)
    print("\n=== Custom Metrics Summary ===")
    for name, value in summary.items():
        print(f"  {name}: {value:.3f}")

    if should_update_baseline():
        baseline = load_baseline()
        baseline["custom_metrics"] = summary
        save_baseline(baseline)
        print("\nBaseline updated (custom_metrics).")
    else:
        baseline = load_baseline().get("custom_metrics", {})
        failures = check_regression(summary, {"metrics": baseline})
        if failures:
            print("\nRegression vs baseline:")
            for line in failures:
                print(f"  - {line}")
            raise SystemExit(1)
