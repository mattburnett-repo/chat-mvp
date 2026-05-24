"""
RAGAS evaluation — faithfulness, context precision/recall, answer relevancy/correctness.

🔍 RAGEvaluator components:

Key Components:
• Metric Selection: Choose which RAGAS metrics to evaluate (defaults to comprehensive set)
• Dataset Preparation: Format test cases into RAGAS-compatible structure
• Evaluation Pipeline: Run metrics and collect results
• Report Generation: Create human-readable summaries with statistics

Test Case Format:
• question: The user query
• generated_answer: Your RAG system's response
• retrieved_contexts: List of chunks used for generation
• ground_truth: Expected correct answer

💡 How the Evaluation Process Works:

1. Prepare Dataset: Convert your test cases into RAGAS Dataset format
2. Run Metrics: RAGAS uses LLMs to evaluate each metric
3. Aggregate Results: Calculate overall and per-question scores
4. Statistical Analysis: Compute mean, std, min, max for each metric
5. Identify Weaknesses: Flag metrics below threshold (0.7 default)
6. Generate Report: Create actionable insights from results

🎯 Deep Dive: Evaluation Report Analysis

Overall Scores:
• Aggregate performance across all test cases
• Higher scores (closer to 1.0) indicate better performance
• Each metric evaluates a different aspect of RAG quality

Metric Analysis:
• Mean: Average performance (main indicator)
• Std: Consistency - lower is better
• Range: Shows worst and best case performance
• Median: Typical performance (less affected by outliers)

🎯 Deep Dive: Metric Calculations

Faithfulness Score:
• Uses NLI (Natural Language Inference) to check if claims in the answer are supported by context
• Score = (Number of supported claims) / (Total claims in answer)
• Range: 0-1, where 1 means fully faithful

Context Precision:
• Evaluates if relevant contexts appear at the top of retrieved results
• Uses reciprocal rank scoring for position-aware evaluation
• Critical for user experience - relevant info should appear first

⚠️ Important Considerations:

• RAGAS metrics require an LLM for evaluation (adds cost)
• Ground truths are needed for recall and some precision metrics
• Results can vary based on the evaluation LLM used
• Consider creating a diverse test set covering edge cases
• Run evaluations periodically to catch regressions
"""

import os
import warnings
from pathlib import Path
from typing import Any, cast

from datasets import Dataset
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.embeddings.huggingface_provider import (
    HuggingFaceEmbeddings as RagasHFEmbeddings,
)
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Chat models (e.g. Llama) are not embedding models; use a small HF embedding model on the same API key
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Embeddings model runs on serverless hf-inference even when the chat model uses another provider
HF_EMBEDDING_PROVIDER = "hf-inference"


class HFEmbeddingsForMetrics(BaseRagasEmbeddings):
    """Bridge modern RAGAS HF embeddings to metrics that call embed_query / embed_documents."""

    def __init__(self, inner: RagasHFEmbeddings):
        super().__init__()
        self._inner = inner
        self.set_run_config(RunConfig())

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._inner.aembed_text(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.aembed_texts(texts)


def _primary_llm_settings() -> tuple[str, str, str]:
    model = (os.getenv("PRIMARY_LLM_MODEL") or "").strip()
    api_key = (os.getenv("PRIMARY_LLM_KEY") or "").strip()
    # Llama 3.1 8B is not on hf-inference; use auto (or nscale/novita) — see model page on the Hub
    provider = (os.getenv("PRIMARY_LLM_PROVIDER") or "auto").strip().lower()
    if not model or not api_key:
        raise ValueError("Set PRIMARY_LLM_MODEL and PRIMARY_LLM_KEY in .env")
    return model, api_key, provider


def get_primary_llm() -> ChatHuggingFace:
    """Cloud Hugging Face chat model (provider from PRIMARY_LLM_PROVIDER)."""
    model, api_key, provider = _primary_llm_settings()
    if not api_key.startswith("hf_"):
        raise ValueError("PRIMARY_LLM_KEY must be a Hugging Face token (hf_...)")
    llm = HuggingFaceEndpoint(
        model=model,
        task="text-generation",
        provider=provider,
        temperature=0,
        do_sample=False,
        huggingfacehub_api_token=api_key,
        max_new_tokens=1024,  # answer_correctness emits long JSON; 512 caused LLMDidNotFinishException
    )
    return ChatHuggingFace(llm=llm)


def get_primary_embeddings() -> HFEmbeddingsForMetrics:
    """Cloud Hugging Face embeddings (hf-inference), LangChain-compatible for RAGAS metrics."""
    from huggingface_hub import InferenceClient

    _, api_key, _ = _primary_llm_settings()
    inner = RagasHFEmbeddings(
        model=HF_EMBEDDING_MODEL,
        use_api=True,
        api_key=api_key,
    )
    # Embedding model is on hf-inference; do not reuse the chat model's provider (e.g. auto).
    inner.client = InferenceClient(
        model=HF_EMBEDDING_MODEL,
        token=api_key,
        provider=HF_EMBEDDING_PROVIDER,
    )
    return HFEmbeddingsForMetrics(inner)


def _metric_name(metric) -> str:
    """RAGAS metric instances use .name, not .__name__."""
    return getattr(metric, "name", metric.__class__.__name__)


def _format_score(score: Any) -> str:
    if isinstance(score, (int, float)):
        return f"{score:.3f}"
    if isinstance(score, list):
        nums = [s for s in score if isinstance(s, (int, float))]
        return f"{sum(nums) / len(nums):.3f}" if nums else "n/a"
    return str(score)


class RAGEvaluator:
    """Run RAGAS metrics over prepared test cases."""

    def __init__(self, metrics=None, llm=None, embeddings=None):
        self.metrics = metrics or [
            context_precision,
            context_recall,
            answer_relevancy,
            faithfulness,
            answer_correctness,
        ]
        self.llm = llm if llm is not None else get_primary_llm()
        self.embeddings = (
            embeddings if embeddings is not None else get_primary_embeddings()
        )

    def prepare_evaluation_dataset(self, test_cases: list[dict[str, Any]]) -> Dataset:
        """Prepare dataset for RAGAS evaluation (RAGAS 0.4+ column names)."""
        evaluation_data = {
            "user_input": [],
            "response": [],
            "retrieved_contexts": [],
            "reference": [],
        }

        for case in test_cases:
            evaluation_data["user_input"].append(case["question"])
            evaluation_data["response"].append(case["generated_answer"])
            evaluation_data["retrieved_contexts"].append(case["retrieved_contexts"])
            evaluation_data["reference"].append(case["ground_truth"])

        return Dataset.from_dict(evaluation_data)

    def evaluate_rag_system(self, test_cases: list[dict[str, Any]]):
        """Run comprehensive RAG evaluation"""
        dataset = self.prepare_evaluation_dataset(test_cases)

        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        df = cast(Any, results).to_pandas()

        evaluation_report = {
            "overall_scores": {},
            "per_question_scores": [],
            "metric_analysis": {},
        }

        for metric in self.metrics:
            metric_name = _metric_name(metric)
            evaluation_report["overall_scores"][metric_name] = df[metric_name].mean()

        for i in range(len(test_cases)):
            row = df.iloc[i]
            question_scores = {
                "question": test_cases[i]["question"],
                "scores": {},
            }
            for metric in self.metrics:
                metric_name = _metric_name(metric)
                question_scores["scores"][metric_name] = row[metric_name]

            evaluation_report["per_question_scores"].append(question_scores)

        for metric in self.metrics:
            metric_name = _metric_name(metric)
            scores = df[metric_name].dropna()

            evaluation_report["metric_analysis"][metric_name] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "min": scores.min(),
                "max": scores.max(),
                "median": scores.median(),
            }

        return evaluation_report

    def generate_evaluation_report(self, results: dict):
        """Generate human-readable evaluation report"""
        report = []
        report.append("=== RAG System Evaluation Report ===\n")

        report.append("Overall Scores:")
        for metric, score in results["overall_scores"].items():
            report.append(f"  {metric}: {_format_score(score)}")

        report.append("\nMetric Analysis:")
        for metric, stats in results["metric_analysis"].items():
            report.append(f"\n{metric}:")
            report.append(f"  Mean: {stats['mean']:.3f} (±{stats['std']:.3f})")
            report.append(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            report.append(f"  Median: {stats['median']:.3f}")

        report.append("\nAreas for Improvement:")
        weak_metrics = []
        for metric, score in results["overall_scores"].items():
            if isinstance(score, float) and score != score:  # nan
                weak_metrics.append((metric, score))
            elif isinstance(score, (int, float)) and score < 0.7:
                weak_metrics.append((metric, score))

        if weak_metrics:
            for metric, score in sorted(
                weak_metrics,
                key=lambda x: (
                    0 if isinstance(x[1], float) and x[1] != x[1] else 1,
                    x[1],
                ),
            ):
                if isinstance(score, float) and score != score:
                    report.append(f"  ⚠️ {metric}: failed (no score)")
                else:
                    report.append(f"  ⚠️ {metric}: {score:.3f}")
        else:
            report.append("  ✅ All metrics above threshold!")

        return "\n".join(report)


if __name__ == "__main__":
    evaluator = RAGEvaluator()

    test_cases = [
        {
            "question": "What is the capital of France?",
            "generated_answer": "The capital of France is Paris.",
            "retrieved_contexts": [
                "Paris is the capital and largest city of France.",
                "France is a country in Western Europe.",
            ],
            "ground_truth": "Paris is the capital of France.",
        },
        {
            "question": "Explain photosynthesis process",
            "generated_answer": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "retrieved_contexts": [
                "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
                "During photosynthesis, plants absorb carbon dioxide and release oxygen.",
            ],
            "ground_truth": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
        },
    ]

    results = evaluator.evaluate_rag_system(test_cases)
    report = evaluator.generate_evaluation_report(results)
    print(report)
