"""
🔍 Understanding Custom RAG Metrics:

Core Custom Metrics:
• Hallucination Score: Detects if the answer contains unsupported claims
• Completeness Score: Measures if all expected aspects are covered
• Response Time Score: Evaluates latency performance
• Citation Accuracy: Verifies correct source attribution

Why Custom Metrics Matter:
• Domain-specific requirements need tailored evaluation
• Standard metrics may miss critical business needs
• Custom metrics enable precise optimization

💡 How Custom Metrics Work:

Hallucination Detection:
1. Split answer into factual claims (sentences)
2. Check each claim against retrieved context using embeddings
3. Calculate similarity scores to determine support
4. Return percentage of supported claims

Completeness Scoring:
1. Define expected aspects for the query type
2. Check direct mentions and semantic similarity
3. Calculate coverage percentage
4. Higher scores indicate comprehensive answers

🎯 Deep Dive: Embedding-Based Validation

Semantic Similarity Checking:
• Uses sentence transformers for dense embeddings
• Cosine similarity measures semantic closeness
• Threshold of 0.8 for "supported" claims
• Lower thresholds may allow hallucinations

Response Time Scoring:
• Target time: 2000ms (configurable)
• Perfect score (1.0) if under target
• Exponential decay for slower responses
• Balances user experience with quality

⚠️ Custom Metric Considerations:

• Embedding models have their own biases and limitations
• Thresholds need tuning for your specific domain
• Citation patterns must match your formatting style
• Consider computational cost of complex metrics
• Validate custom metrics against human judgment
"""

# How to create custom evaluation metrics for LangChain RAG systems
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Local embeddings (needs torch + sentence-transformers — not available on this env):
# from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_PROVIDER = "hf-inference"


class CustomRAGMetrics:
    """Learn how to build custom metrics for evaluating LangChain RAG - tutorial"""

    def __init__(self):
        # self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        api_key = (os.getenv("PRIMARY_LLM_KEY") or "").strip()
        if not api_key:
            raise ValueError("Set PRIMARY_LLM_KEY in .env")
        self._client = InferenceClient(
            model=EMBEDDING_MODEL,
            token=api_key,
            provider=EMBEDDING_PROVIDER,
        )

    def _encode_one(self, text: str) -> np.ndarray:
        resp = self._client.feature_extraction(text)
        vec = resp[0] if isinstance(resp[0], list) else resp
        return np.array(vec, dtype=float)

    def _encode_many(self, texts: list[str]) -> np.ndarray:
        return np.array([self._encode_one(t) for t in texts])

    def hallucination_score(self, answer: str, contexts: list[str]) -> float:
        """Measure how much the answer hallucinates beyond context"""
        answer_sentences = answer.split(".")
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]

        combined_context = " ".join(contexts)

        hallucination_count = 0
        for sentence in answer_sentences:
            if not self._is_supported_by_context(sentence, combined_context):
                hallucination_count += 1

        if len(answer_sentences) == 0:
            return 1.0

        return 1.0 - (hallucination_count / len(answer_sentences))

    def _is_supported_by_context(self, claim: str, context: str) -> bool:
        """Check if a claim is supported by context using embeddings"""
        # claim_embedding = self.sentence_model.encode(claim)
        claim_embedding = self._encode_one(claim)

        context_sentences = context.split(".")
        context_sentences = [s.strip() for s in context_sentences if s.strip()]

        if not context_sentences:
            return False

        # context_embeddings = self.sentence_model.encode(context_sentences)
        context_embeddings = self._encode_many(context_sentences)

        similarities = np.dot(context_embeddings, claim_embedding) / (
            np.linalg.norm(context_embeddings, axis=1) * np.linalg.norm(claim_embedding)
        )

        return np.max(similarities) > 0.8

    def completeness_score(
        self, answer: str, question: str, expected_aspects: list[str]
    ) -> float:
        """Measure if answer covers all expected aspects"""
        covered_aspects = 0
        answer_lower = answer.lower()

        for aspect in expected_aspects:
            if aspect.lower() in answer_lower:
                covered_aspects += 1
            else:
                # aspect_embedding = self.sentence_model.encode(aspect)
                # answer_embedding = self.sentence_model.encode(answer)
                aspect_embedding = self._encode_one(aspect)
                answer_embedding = self._encode_one(answer)

                similarity = np.dot(aspect_embedding, answer_embedding) / (
                    np.linalg.norm(aspect_embedding) * np.linalg.norm(answer_embedding)
                )

                if similarity > 0.7:
                    covered_aspects += 1

        return covered_aspects / len(expected_aspects) if expected_aspects else 1.0

    def response_time_score(
        self, response_time_ms: float, target_time_ms: float = 2000
    ) -> float:
        """Score based on response time performance"""
        if response_time_ms <= target_time_ms:
            return 1.0
        return np.exp(-0.0005 * (response_time_ms - target_time_ms))

    def citation_accuracy(
        self, answer: str, contexts: list[str], citations: list[int]
    ) -> float:
        """Measure if citations correctly reference source contexts"""
        citation_pattern = r"\[(\d+)\]"
        sentences_with_citations = []

        for sentence in answer.split("."):
            if re.search(citation_pattern, sentence):
                citations_in_sentence = [
                    int(c) for c in re.findall(citation_pattern, sentence)
                ]
                sentences_with_citations.append((sentence, citations_in_sentence))

        if not sentences_with_citations:
            return 1.0 if not citations else 0.0

        correct_citations = 0
        total_citations = 0

        for sentence, cited_indices in sentences_with_citations:
            clean_sentence = re.sub(citation_pattern, "", sentence).strip()

            for idx in cited_indices:
                total_citations += 1
                if 0 <= idx < len(contexts):
                    if self._is_supported_by_context(clean_sentence, contexts[idx]):
                        correct_citations += 1

        return correct_citations / total_citations if total_citations > 0 else 1.0


# Example usage
custom_metrics = CustomRAGMetrics()

answer = "Paris is the capital of France. It has a population of 2.2 million. The Eiffel Tower was built in 1789."
contexts = [
    "Paris is the capital city of France.",
    "The Eiffel Tower is a famous landmark in Paris, completed in 1889.",
]

hallucination_score = custom_metrics.hallucination_score(answer, contexts)
print(f"Hallucination Score: {hallucination_score:.3f}")

question = "What are the main features of Python?"
answer = "Python is a high-level programming language known for its simplicity and readability."
expected_aspects = [
    "high-level",
    "interpreted",
    "dynamic typing",
    "readability",
    "large ecosystem",
]

completeness = custom_metrics.completeness_score(answer, question, expected_aspects)
print(f"Completeness Score: {completeness:.3f}")

response_score = custom_metrics.response_time_score(1500)
print(f"Response Time Score: {response_score:.3f}")
