"""
🔍 Understanding RAG Cost Optimization:

Cost Components:
• Embedding Costs: One-time cost for document processing
• Retrieval Costs: Vector database query expenses
• Generation Costs: LLM API costs (input + output tokens)
• Infrastructure: Hosting, storage, compute resources

Optimization Strategies:
• Model selection (GPT-3.5 vs GPT-4)
• Context window management
• Response caching
• Batch processing

💡 How Cost Analysis Works:

1. Usage Profiling: Analyze query patterns and volumes
2. Cost Breakdown: Calculate costs per component
3. Token Estimation: Convert text to approximate tokens
4. Monthly Projection: Scale to expected usage
5. Optimization Identification: Find cost reduction opportunities
6. Priority Ranking: Sort by impact and ease of implementation

🎯 Deep Dive: Cost-Quality Tradeoffs

Model Selection Impact:
• GPT-4: Higher quality, 20x more expensive
• GPT-3.5-turbo: Good quality, cost-effective
• Quality threshold determines viable options

Context Optimization:
• Fewer chunks = lower cost
• Smaller chunks = better precision
• Sweet spot: 3-5 chunks of 500 tokens
• Reranking helps maintain quality with fewer chunks

⚠️ Cost Optimization Warnings:

• Don't sacrifice critical quality for minor savings
• Cache invalidation strategy is crucial
• Monitor quality metrics after optimizations
• Consider user segments - some may need premium quality
• Factor in hidden costs (development, maintenance)

===

How estimates are generated (criteria only):    

hard-coded around line 313 in this file:
    usage_stats = {
        "monthly_queries": 50000,
        "avg_doc_length": 1500,
        "avg_chunks_retrieved": 5,
        "avg_response_length": 300,
    }

"""

# How to optimize RAG costs in LangChain - complete implementation guide
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from corpus.embeddings import embedding_model_name

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Legacy cloud cost table (OpenAI / Gemini) — kept for reference:
# "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
# "gpt-4": {"input": 0.03, "output": 0.06},
# "gemini-2.0-flash": {"input": 0.00075, "output": 0.0015},


def _default_hf_model_id() -> str:
    return (
        os.getenv("PRIMARY_LLM_MODEL") or "meta-llama/Llama-3.1-8B-Instruct"
    ).strip()


def _hf_cost_rates(model: str) -> dict[str, float]:
    """Nominal $/1K tokens for HF Inference (placeholders for comparison only)."""
    model_lower = model.lower()
    if "llama" in model_lower or "meta-llama" in model_lower:
        return {"input": 0.0002, "output": 0.0002}
    if "mistral" in model_lower:
        return {"input": 0.00025, "output": 0.00025}
    return {"input": 0.0003, "output": 0.0003}


class RAGCostOptimizer:
    """Learn how to analyze and optimize LangChain RAG costs step-by-step (Hugging Face)."""

    def __init__(self):
        self.embedding_model = embedding_model_name()
        # HF embedding via hf-inference (per 1K tokens, placeholder)
        self.embedding_rate_per_1k = 0.0001
        self.retrieval_rate_per_1k = (
            0.0  # local/pgvector retrieval has no per-token API cost
        )

    def analyze_cost_breakdown(
        self, rag_config: dict[str, Any], usage_stats: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze cost breakdown for a RAG system using HF-hosted models."""
        monthly_queries = usage_stats.get("monthly_queries", 10000)
        avg_doc_length = usage_stats.get("avg_doc_length", 1000)
        avg_chunks_retrieved = usage_stats.get("avg_chunks_retrieved", 5)
        model = rag_config.get("model") or _default_hf_model_id()

        embedding_cost = self._calculate_embedding_cost(monthly_queries, avg_doc_length)

        retrieval_cost = self._calculate_retrieval_cost(
            monthly_queries, avg_chunks_retrieved, avg_doc_length
        )

        generation_cost = self._calculate_generation_cost(
            model,
            monthly_queries,
            avg_chunks_retrieved * avg_doc_length,
            usage_stats.get("avg_response_length", 200),
        )

        total_monthly = embedding_cost + retrieval_cost + generation_cost

        return {
            "breakdown": {
                "embedding": embedding_cost,
                "retrieval": retrieval_cost,
                "generation": generation_cost,
            },
            "total_monthly": total_monthly,
            "cost_per_query": total_monthly / monthly_queries
            if monthly_queries
            else 0.0,
            "model": model,
            "embedding_model": self.embedding_model,
            "optimization_suggestions": self._generate_cost_optimizations(
                rag_config, usage_stats, total_monthly
            ),
        }

    def _calculate_embedding_cost(self, queries: int, doc_length: int) -> float:
        """HF embedding API cost (hf-inference)."""
        new_embeddings = queries * 0.2
        tokens = (new_embeddings * doc_length) / 4
        return (tokens / 1000) * self.embedding_rate_per_1k

    def _calculate_retrieval_cost(
        self, queries: int, chunks_retrieved: int, doc_length: int
    ) -> float:
        """Retrieval cost — zero for self-hosted DB; optional compute proxy."""
        if self.retrieval_rate_per_1k <= 0:
            return 0.0
        tokens = (queries * chunks_retrieved * doc_length) / 4
        return (tokens / 1000) * self.retrieval_rate_per_1k

    def _calculate_generation_cost(
        self,
        model: str,
        queries: int,
        context_chars: int,
        response_chars: int,
    ) -> float:
        """HF chat / Inference Providers generation cost estimate."""
        input_tokens = (context_chars * queries) / 4
        output_tokens = (response_chars * queries) / 4

        model_costs = _hf_cost_rates(model)

        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]

        return input_cost + output_cost

    def _generate_cost_optimizations(
        self,
        rag_config: dict[str, Any],
        usage_stats: dict[str, Any],
        total_monthly: float,
    ) -> list[str]:
        """High-level cost tips based on config and spend."""
        suggestions: list[str] = []
        model = rag_config.get("model") or _default_hf_model_id()
        provider = (os.getenv("PRIMARY_LLM_PROVIDER") or "auto").strip().lower()

        if total_monthly > 100:
            suggestions.append(
                "High monthly spend: reduce retrieval_k or enable response caching for repeat queries."
            )
        if rag_config.get("retrieval_k", 5) > 3:
            suggestions.append(
                "Lower retrieval_k to 3 to cut input tokens to the chat model."
            )
        if rag_config.get("chunk_size", 1000) > 500:
            suggestions.append(
                "Use ~500-token chunks to reduce context size per query."
            )
        if provider == "auto":
            suggestions.append(
                "Chat uses Inference Providers (auto); embeddings use hf-inference — track both quotas."
            )
        if "8b" not in model.lower() and "small" not in model.lower():
            suggestions.append(
                f"Consider a smaller HF model than {model} if quality allows (e.g. 8B instruct)."
            )
        if not suggestions:
            suggestions.append(
                "Configuration looks reasonable for the given usage assumptions."
            )
        return suggestions

    def optimize_configuration(
        self,
        current_config: dict[str, Any],
        quality_threshold: float = 0.8,
    ) -> dict[str, Any]:
        """Suggest optimal configuration for cost-quality balance (Hugging Face)."""
        optimizations: list[dict[str, Any]] = []
        model = current_config.get("model") or _default_hf_model_id()
        hf_default = _default_hf_model_id()

        if model != hf_default and quality_threshold < 0.9:
            optimizations.append(
                {
                    "action": f"Use default HF model {hf_default} from .env for lower cost",
                    "impact": "Depends on task; often small quality tradeoff",
                    "savings": 0.3,
                }
            )

        if current_config.get("chunk_size", 1000) > 500:
            optimizations.append(
                {
                    "action": "Reduce chunk size to 500 tokens",
                    "impact": "Better precision, ~20% less context tokens",
                    "savings": 0.2,
                }
            )

        if current_config.get("retrieval_k", 5) > 3:
            optimizations.append(
                {
                    "action": "Reduce retrieval_k to 3",
                    "impact": "~40% less context, minimal quality impact for many queries",
                    "savings": 0.4,
                }
            )

        optimizations.append(
            {
                "action": "Cache HF responses for common queries (disk or Redis)",
                "impact": "Large savings on repeated prompts",
                "savings": 0.6,
            }
        )

        optimizations.append(
            {
                "action": "Run embeddings on hf-inference; keep chat on auto only when needed",
                "impact": "Avoids routing embedding calls through paid provider router",
                "savings": 0.15,
            }
        )

        return {
            "optimizations": optimizations,
            "estimated_savings": sum(opt["savings"] for opt in optimizations[:3]) / 3,
            "implementation_priority": self._prioritize_optimizations(optimizations),
        }

    def _prioritize_optimizations(
        self, optimizations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prioritize optimizations by impact and ease."""
        for opt in optimizations:
            if "caching" in opt["action"].lower():
                opt["priority_score"] = opt["savings"] * 0.8
            elif "model" in opt["action"].lower():
                opt["priority_score"] = opt["savings"] * 1.0
            else:
                opt["priority_score"] = opt["savings"] * 0.9

        return sorted(optimizations, key=lambda x: x["priority_score"], reverse=True)


if __name__ == "__main__":
    optimizer = RAGCostOptimizer()

    current_config = {
        "model": _default_hf_model_id(),
        "chunk_size": 1000,
        "retrieval_k": 5,
        "provider": (os.getenv("PRIMARY_LLM_PROVIDER") or "auto").strip().lower(),
    }

    usage_stats = {
        "monthly_queries": 50000,
        "avg_doc_length": 1500,
        "avg_chunks_retrieved": 5,
        "avg_response_length": 300,
    }

    cost_analysis = optimizer.analyze_cost_breakdown(current_config, usage_stats)
    monthly_cost = cost_analysis.get("total_monthly", 0)
    cost_per_query = cost_analysis.get("cost_per_query", 0)

    print(f"Model: {cost_analysis.get('model')}")
    print(f"Embeddings: {cost_analysis.get('embedding_model')}")
    print(f"Monthly cost (estimate): ${monthly_cost:.2f}")
    print(f"Cost per query (estimate): ${cost_per_query:.4f}")

    print("\nCost breakdown:")
    for component, amount in cost_analysis.get("breakdown", {}).items():
        print(f"  {component}: ${amount:.2f}")

    print("\nTips:")
    for tip in cost_analysis.get("optimization_suggestions", []):
        print(f"  - {tip}")

    optimizations = optimizer.optimize_configuration(current_config)
    print("\nOptimization suggestions:")
    for opt in optimizations.get("optimizations", []):
        action = opt.get("action", "")
        savings_pct = int(opt.get("savings", 0) * 100)
        print(f"  - {action}: ~{savings_pct}% potential savings")
