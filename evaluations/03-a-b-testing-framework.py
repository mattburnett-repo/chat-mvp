"""
🔍 Understanding RAG A/B Testing:

Key Components:
• RAGVariant: Configuration dataclass defining test parameters
• Test Registration: Register multiple RAG system configurations
• Metric Tracking: Collect performance, quality, and cost metrics
• Statistical Analysis: Determine significant differences between variants

What to Test:
• Chunk sizes and overlap settings
• Different embedding models
• Retrieval strategies (k values, reranking)
• LLM models and temperatures
• Prompt templates and instructions

💡 How A/B Testing Works:

1. Variant Setup: Define different RAG configurations to test
2. Query Distribution: Run same queries through all variants
3. Metric Collection: Track response time, quality scores, costs
4. Statistical Testing: Use t-tests to find significant differences
5. Effect Size Analysis: Measure practical significance
6. Recommendation Generation: Identify best performing variant

🎯 Deep Dive: Statistical Significance

T-Test Analysis:
• Compares means between two variants
• P-value < 0.05 indicates significant difference
• Effect size shows practical importance
• Cohen's d: 0.2=small, 0.5=medium, 0.8=large

Composite Scoring:
• Combines multiple metrics into single score
• Weight factors based on business priorities
• Balance quality, performance, and cost
• Customize formula for your use case

⚠️ A/B Testing Best Practices:

• Test one major change at a time for clarity
• Ensure sufficient sample size for statistical power
• Run tests long enough to capture variance
• Consider time-of-day and user segment effects
• Always validate results with real user feedback
• Document configuration differences clearly
"""

# How to implement A/B testing for LangChain RAG systems - complete tutorial
import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from scipy import stats

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_PROVIDER = "hf-inference"

# Legacy cloud cost table (OpenAI / Gemini) — kept for reference:
# cost_per_1k = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.03, "gemini-2.0-flash": 0.001}


def _primary_llm_settings() -> tuple[str, str, str]:
    model = (os.getenv("PRIMARY_LLM_MODEL") or "").strip()
    api_key = (os.getenv("PRIMARY_LLM_KEY") or "").strip()
    provider = (os.getenv("PRIMARY_LLM_PROVIDER") or "auto").strip().lower()
    if not model or not api_key:
        raise ValueError("Set PRIMARY_LLM_MODEL and PRIMARY_LLM_KEY in .env")
    return model, api_key, provider


def get_hf_chat_llm(
    temperature: float = 0.0, max_new_tokens: int = 512
) -> ChatHuggingFace:
    """Hugging Face chat LLM for A/B variants (provider from .env)."""
    hf_model, api_key, provider = _primary_llm_settings()
    if not api_key.startswith("hf_"):
        raise ValueError("PRIMARY_LLM_KEY must be a Hugging Face token (hf_...)")
    llm = HuggingFaceEndpoint(
        model=hf_model,
        task="text-generation",
        provider=provider,
        temperature=temperature,
        do_sample=False,
        huggingfacehub_api_token=api_key,
        max_new_tokens=max_new_tokens,
    )
    return ChatHuggingFace(llm=llm)


class HFEmbeddingScorer:
    """Semantic relevancy via HF Inference API (no local torch)."""

    def __init__(self):
        api_key = (os.getenv("PRIMARY_LLM_KEY") or "").strip()
        self._client = InferenceClient(
            model=HF_EMBEDDING_MODEL,
            token=api_key,
            provider=HF_EMBEDDING_PROVIDER,
        )

    def _encode_one(self, text: str) -> np.ndarray:
        resp = self._client.feature_extraction(text)
        vec = resp[0] if isinstance(resp[0], list) else resp
        return np.array(vec, dtype=float)

    def similarity(self, text_a: str, text_b: str) -> float:
        a, b = self._encode_one(text_a), self._encode_one(text_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


PROMPT_TEMPLATES = {
    "standard": (
        "Answer the question using only the context below. Be concise.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "enhanced": (
        "You are a careful assistant. Use only the provided context. "
        "If the context is insufficient, say so briefly.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
}


class SimpleHFRAGSystem:
    """Minimal RAG stand-in for A/B demos: retrieve from provided contexts, answer with HF LLM."""

    def __init__(self, config: "RAGVariant"):
        self.config = config
        self.llm = get_hf_chat_llm(
            temperature=config.temperature,
            max_new_tokens=512,
        )
        self._embed_scorer = HFEmbeddingScorer()

    def query(
        self,
        question: str,
        k: int = 5,
        contexts: list[str] | None = None,
    ) -> dict[str, Any]:
        contexts = contexts or []
        # Simulate retrieval: take first k chunks (real app would use vector search)
        retrieved = contexts[:k]
        template = PROMPT_TEMPLATES.get(
            self.config.prompt_template, PROMPT_TEMPLATES["standard"]
        )
        context_block = "\n\n".join(retrieved) if retrieved else "(no context provided)"
        prompt = template.format(context=context_block, question=question)

        response = self.llm.invoke(
            [
                SystemMessage(content="Answer from context only."),
                HumanMessage(content=prompt),
            ]
        )
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "source_documents": retrieved,
        }


@dataclass
class RAGVariant:
    """Configuration for a RAG variant"""

    name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    reranking: bool
    model: str  # HF repo id or label; defaults from PRIMARY_LLM_MODEL in examples
    temperature: float
    prompt_template: str


class RAGABTester:
    """Learn how to A/B test LangChain RAG systems step-by-step"""

    def __init__(self, embedding_scorer: HFEmbeddingScorer | None = None):
        self.test_results = []
        self.variants: dict[str, dict[str, Any]] = {}
        self._embed_scorer = embedding_scorer or HFEmbeddingScorer()

    def register_variant(
        self,
        variant_name: str,
        rag_system: Any,
        config: RAGVariant,
    ):
        """Register a RAG variant for testing"""
        self.variants[variant_name] = {
            "system": rag_system,
            "config": config,
            "results": [],
        }

    def run_test(
        self,
        test_queries: list[dict[str, Any]],
        metrics_to_track: list[str] | None = None,
    ):
        """Run A/B test across all variants"""
        metrics_to_track = metrics_to_track or [
            "response_time",
            "relevancy_score",
            "faithfulness",
            "cost",
            "user_satisfaction",
        ]

        print(f"Running A/B test with {len(test_queries)} queries...")
        print(f"Testing {len(self.variants)} variants: {list(self.variants.keys())}")

        for query_data in test_queries:
            query_id = hashlib.md5(query_data["query"].encode()).hexdigest()[:8]

            for variant_name, variant_data in self.variants.items():
                result = self._test_variant(
                    variant_name,
                    variant_data,
                    query_data,
                    query_id,
                )
                variant_data["results"].append(result)

        return self._analyze_results(metrics_to_track)

    def _test_variant(
        self,
        variant_name: str,
        variant_data: dict,
        query_data: dict,
        query_id: str,
    ) -> dict:
        """Test a single variant with a query"""
        rag_system = variant_data["system"]
        config = variant_data["config"]

        start_time = time.time()

        try:
            contexts = query_data.get("contexts", [])
            response = rag_system.query(
                query_data["query"],
                k=config.retrieval_k,
                contexts=contexts,
            )

            response_time = (time.time() - start_time) * 1000

            metrics: dict[str, Any] = {
                "variant": variant_name,
                "query_id": query_id,
                "response_time": response_time,
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

            if "expected_answer" in query_data:
                metrics["relevancy_score"] = self._calculate_relevancy(
                    response["answer"],
                    query_data["expected_answer"],
                )

            metrics["cost"] = self._estimate_cost(
                config.model,
                len(response.get("answer", "")),
                len(str(response.get("source_documents", []))),
            )

            metrics["user_satisfaction"] = self._simulate_user_satisfaction(
                metrics.get("relevancy_score", 0),
                response_time,
            )

        except Exception as e:
            metrics = {
                "variant": variant_name,
                "query_id": query_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        return metrics

    def _calculate_relevancy(self, answer: str, expected: str) -> float:
        """Semantic relevancy via HF embeddings API."""
        try:
            return self._embed_scorer.similarity(answer, expected)
        except Exception:
            from difflib import SequenceMatcher

            return SequenceMatcher(None, answer.lower(), expected.lower()).ratio()

    def _estimate_cost(
        self, model: str, answer_chars: int, context_chars: int
    ) -> float:
        """Rough HF cost proxy (tokens ≈ chars/4). Tune or replace with real billing data."""
        total_tokens = (answer_chars + context_chars) / 4
        # Nominal $/1K tokens for hosted inference (placeholder for comparison only)
        cost_per_1k = 0.0002 if "llama" in model.lower() else 0.0005
        return (total_tokens / 1000) * cost_per_1k

    def _simulate_user_satisfaction(
        self, relevancy: float, response_time: float
    ) -> float:
        """Simulate user satisfaction score"""
        time_factor = 1.0 if response_time < 2000 else 0.8
        return min(1.0, relevancy * time_factor + np.random.normal(0, 0.1))

    def _analyze_results(self, metrics: list[str]) -> dict:
        """Analyze A/B test results with statistical significance"""
        analysis: dict[str, Any] = {
            "summary": {},
            "statistical_tests": {},
            "recommendations": [],
        }

        all_results = []
        for variant_name, variant_data in self.variants.items():
            for result in variant_data["results"]:
                result["variant_name"] = variant_name
                all_results.append(result)

        df = pd.DataFrame(all_results)

        for variant in self.variants.keys():
            variant_df = df[df["variant_name"] == variant]

            analysis["summary"][variant] = {
                "total_queries": len(variant_df),
                "success_rate": variant_df["success"].mean()
                if "success" in variant_df
                else 0,
                "metrics": {},
            }

            for metric in metrics:
                if metric in variant_df.columns:
                    metric_data = variant_df.loc[:, metric].dropna()
                    if len(metric_data) > 0:
                        analysis["summary"][variant]["metrics"][metric] = {
                            "mean": metric_data.mean(),
                            "std": metric_data.std(),
                            "median": metric_data.median(),
                            "min": metric_data.min(),
                            "max": metric_data.max(),
                        }

        if len(self.variants) == 2:
            variants = list(self.variants.keys())
            for metric in metrics:
                if metric in df.columns:
                    group1 = df.loc[df["variant_name"] == variants[0], metric].dropna()
                    group2 = df.loc[df["variant_name"] == variants[1], metric].dropna()

                    if len(group1) > 1 and len(group2) > 1:
                        t_stat, p_value_raw = stats.ttest_ind(group1, group2)
                        p_value = cast(float, p_value_raw)

                        analysis["statistical_tests"][metric] = {
                            "test": "independent_t_test",
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "effect_size": (group1.mean() - group2.mean())
                            / np.sqrt((group1.std() ** 2 + group2.std() ** 2) / 2),
                        }

        analysis["recommendations"] = self._generate_recommendations(analysis)
        return analysis

    def _generate_recommendations(self, analysis: dict) -> list[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        best_variant = None
        best_score = -float("inf")

        for variant, data in analysis["summary"].items():
            if "metrics" in data:
                score = 0
                if "relevancy_score" in data["metrics"]:
                    score += data["metrics"]["relevancy_score"]["mean"] * 2
                if "response_time" in data["metrics"]:
                    score -= data["metrics"]["response_time"]["mean"] / 10000
                if "cost" in data["metrics"]:
                    score -= data["metrics"]["cost"]["mean"] * 10

                if score > best_score:
                    best_score = score
                    best_variant = variant

        if best_variant:
            recommendations.append(
                f"Variant '{best_variant}' shows the best overall performance"
            )

        for metric, test_data in analysis["statistical_tests"].items():
            if test_data["significant"]:
                effect = "large" if abs(test_data["effect_size"]) > 0.8 else "moderate"
                recommendations.append(
                    f"Significant difference in {metric} (p={test_data['p_value']:.3f}, "
                    f"{effect} effect size)"
                )

        return recommendations

    def print_report(self, analysis: dict) -> None:
        """Print human-readable A/B summary"""
        print("\n=== A/B Test Report ===\n")
        for variant, data in analysis["summary"].items():
            print(f"{variant}:")
            print(
                f"  queries: {data['total_queries']}, success: {data['success_rate']:.2%}"
            )
            for metric, metric_stats in data.get("metrics", {}).items():
                print(
                    f"  {metric}: mean={metric_stats['mean']:.3f} "
                    f"(±{metric_stats['std']:.3f})"
                )
        print("\nRecommendations:")
        for line in analysis["recommendations"]:
            print(f"  - {line}")


def _default_hf_model_id() -> str:
    return (
        os.getenv("PRIMARY_LLM_MODEL") or "meta-llama/Llama-3.1-8B-Instruct"
    ).strip()


if __name__ == "__main__":
    hf_model = _default_hf_model_id()
    ab_tester = RAGABTester()

    variant_a = RAGVariant(
        name="baseline",
        chunk_size=1000,
        chunk_overlap=200,
        retrieval_k=3,
        reranking=False,
        model=hf_model,
        temperature=0.3,
        prompt_template="standard",
    )

    variant_b = RAGVariant(
        name="optimized",
        chunk_size=500,
        chunk_overlap=100,
        retrieval_k=5,
        reranking=True,
        model=hf_model,
        temperature=0.1,
        prompt_template="enhanced",
    )

    rag_a = SimpleHFRAGSystem(variant_a)
    rag_b = SimpleHFRAGSystem(variant_b)

    ab_tester.register_variant("baseline", rag_a, variant_a)
    ab_tester.register_variant("optimized", rag_b, variant_b)

    test_queries = [
        {
            "query": "What is the capital of France?",
            "expected_answer": "Paris is the capital of France.",
            "contexts": [
                "Paris is the capital and largest city of France.",
                "France is a country in Western Europe.",
            ],
        },
        {
            "query": "When was the Eiffel Tower completed?",
            "expected_answer": "The Eiffel Tower was completed in 1889.",
            "contexts": [
                "The Eiffel Tower is a famous landmark in Paris, completed in 1889.",
                "Paris is the capital city of France.",
            ],
        },
    ]

    results = ab_tester.run_test(test_queries)
    ab_tester.print_report(results)
