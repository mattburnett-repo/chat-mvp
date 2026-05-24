"""
🔍 Understanding RAG Monitoring:

Core Components:
• Metric Events: Track individual query performance and outcomes
• Alert Thresholds: Define acceptable ranges for key metrics
• Time Windows: Analyze metrics over different periods
• Health Score: Composite metric for overall system status

What to Monitor:
• Performance (latency, throughput)
• Quality (relevancy, faithfulness)
• Reliability (error rates, availability)
• Cost (per-query, daily totals)

💡 How Production Monitoring Works:

1. Event Tracking: Log each query with performance metrics
2. Window Analysis: Calculate statistics over time periods
3. Threshold Checking: Compare metrics against alerts
4. Alert Generation: Trigger notifications for anomalies
5. Dashboard Updates: Real-time visibility into system health
6. Trend Analysis: Identify patterns and degradation

🎯 Deep Dive: Health Score Calculation

Scoring System:
• Start with perfect score (100)
• Deduct points for issues:
- Error rate > 1%: -30 points max
- P95 latency > 2s: -20 points max
- Relevancy < 0.8: -30 points max

Percentile Metrics:
• P50: Median performance
• P95: 95% of requests are faster
• P99: Worst-case for most users

⚠️ Monitoring Best Practices:

• Set realistic alert thresholds to avoid fatigue
• Monitor both technical and business metrics
• Use separate environments for testing changes
• Implement graceful degradation for failures
• Keep historical data for trend analysis
• Correlate metrics with user feedback
"""

# How to implement production monitoring for LangChain RAG - complete tutorial
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_PROVIDER = "hf-inference"


def _primary_llm_settings() -> tuple[str, str, str]:
    model = (os.getenv("PRIMARY_LLM_MODEL") or "").strip()
    api_key = (os.getenv("PRIMARY_LLM_KEY") or "").strip()
    provider = (os.getenv("PRIMARY_LLM_PROVIDER") or "auto").strip().lower()
    if not model or not api_key:
        raise ValueError("Set PRIMARY_LLM_MODEL and PRIMARY_LLM_KEY in .env")
    return model, api_key, provider


def get_hf_chat_llm(
    temperature: float = 0.0, max_new_tokens: int = 256
) -> ChatHuggingFace:
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


def _estimate_hf_query_cost(
    model: str, context_chars: int, response_chars: int
) -> float:
    """Rough HF cost proxy (aligns with optimizeRagCost.py placeholders)."""
    tokens = (context_chars + response_chars) / 4
    rate = 0.0002 if "llama" in model.lower() else 0.0003
    return (tokens / 1000) * rate * 2


@dataclass
class RAGMetricEvent:
    """Single metric event"""

    timestamp: datetime
    metric_type: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGMonitor:
    """Monitor LangChain RAG systems in production (Hugging Face–aware config)."""

    def __init__(
        self,
        alert_thresholds: dict[str, float] | None = None,
        window_size_minutes: int = 5,
        hf_model: str | None = None,
        hf_provider: str | None = None,
    ):
        self.hf_model = hf_model or _primary_llm_settings()[0]
        self.hf_provider = hf_provider or (os.getenv("PRIMARY_LLM_PROVIDER") or "auto")
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_thresholds = (
            alert_thresholds
            or {
                "error_rate": 0.05,
                "p95_latency_ms": 3000,
                "relevancy_score": 0.7,
                "cost_per_query": 0.01,  # HF queries are typically well below OpenAI-scale costs
            }
        )
        self.window_size = timedelta(minutes=window_size_minutes)
        self.alerts: list[dict[str, Any]] = []

        self.logger = logging.getLogger("RAGMonitor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def track_query(
        self,
        query_id: str,
        query: str,
        response: dict[str, Any],
        metrics: dict[str, Any],
    ):
        """Track a single query execution."""
        timestamp = datetime.now()

        self._track_metric("query_count", 1, timestamp)
        self._track_metric(
            "response_time_ms",
            metrics.get("response_time", metrics.get("response_time_ms", 0)),
            timestamp,
        )
        self._track_metric(
            "chunks_retrieved", metrics.get("chunks_retrieved", 0), timestamp
        )
        self._track_metric(
            "relevancy_score", metrics.get("relevancy_score", 0), timestamp
        )
        self._track_metric("cost", metrics.get("cost", 0), timestamp)

        if metrics.get("error"):
            self._track_metric("error_count", 1, timestamp)
            self.logger.error("Query %s failed: %s", query_id, metrics["error"])

        self._check_alerts(timestamp)

    def track_hf_query(
        self,
        query_id: str,
        question: str,
        contexts: list[str],
        llm: ChatHuggingFace,
        expected_answer: str | None = None,
    ) -> dict[str, Any]:
        """Run one HF RAG-style call and record metrics on this monitor."""
        context_block = "\n\n".join(contexts) if contexts else "(no context)"
        prompt = (
            f"Answer using only the context below.\n\nContext:\n{context_block}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        start = time.time()
        error = None
        answer = ""

        try:
            msg = llm.invoke(
                [
                    SystemMessage(content="Answer from context only."),
                    HumanMessage(content=prompt),
                ]
            )
            raw_content = msg.content if hasattr(msg, "content") else str(msg)
            answer = raw_content if isinstance(raw_content, str) else str(raw_content)
        except Exception as exc:
            error = str(exc)

        elapsed_ms = (time.time() - start) * 1000
        relevancy = 0.0
        if expected_answer and answer and not error:
            relevancy = _embedding_similarity(answer, expected_answer)

        cost = _estimate_hf_query_cost(self.hf_model, len(context_block), len(answer))

        payload = {
            "response_time": elapsed_ms,
            "chunks_retrieved": len(contexts),
            "relevancy_score": relevancy,
            "cost": cost,
            "error": error,
        }
        self.track_query(
            query_id,
            question,
            {"answer": answer},
            payload,
        )
        return {"answer": answer, "metrics": payload}

    def _track_metric(self, metric_type: str, value: float, timestamp: datetime):
        event = RAGMetricEvent(
            timestamp=timestamp, metric_type=metric_type, value=value
        )
        self.metrics[metric_type].append(event)

    def _check_alerts(self, current_time: datetime):
        window_start = current_time - self.window_size
        metrics_summary = self._calculate_windowed_metrics(window_start, current_time)

        for metric, threshold in self.alert_thresholds.items():
            if metric not in metrics_summary:
                continue
            value = metrics_summary[metric]

            if metric in ["relevancy_score"]:
                if value < threshold:
                    self._trigger_alert(metric, value, threshold, "below")
            else:
                if value > threshold:
                    self._trigger_alert(metric, value, threshold, "above")

    def _calculate_windowed_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> dict[str, float]:
        summary: dict[str, float] = {}

        query_count = sum(
            1
            for e in self.metrics["query_count"]
            if start_time <= e.timestamp <= end_time
        )
        error_count = sum(
            1
            for e in self.metrics["error_count"]
            if start_time <= e.timestamp <= end_time
        )

        if query_count > 0:
            summary["error_rate"] = error_count / query_count

        response_times = [
            e.value
            for e in self.metrics["response_time_ms"]
            if start_time <= e.timestamp <= end_time
        ]

        if response_times:
            summary["p50_latency_ms"] = float(np.percentile(response_times, 50))
            summary["p95_latency_ms"] = float(np.percentile(response_times, 95))
            summary["p99_latency_ms"] = float(np.percentile(response_times, 99))

        for metric in ["relevancy_score", "cost"]:
            values = [
                e.value
                for e in self.metrics[metric]
                if start_time <= e.timestamp <= end_time
            ]
            if values:
                summary[metric] = float(np.mean(values))
                if metric == "cost" and query_count > 0:
                    summary["cost_per_query"] = summary[metric] / query_count

        return summary

    def _compare_metric_dicts(
        self, current: dict[str, float], baseline: dict[str, float]
    ) -> dict[str, float]:
        trends: dict[str, float] = {}
        for key, cur in current.items():
            prev = baseline.get(key)
            if prev is None:
                continue
            if prev != 0:
                trends[key] = (cur - prev) / abs(prev)
            else:
                trends[key] = 0.0 if cur == 0 else 1.0
        return trends

    def _calculate_trends(self) -> dict[str, float]:
        """Compare recent window vs earlier window; fall back to session half-split for short demos."""
        now = datetime.now()
        recent = self._calculate_windowed_metrics(now - timedelta(minutes=5), now)
        previous = self._calculate_windowed_metrics(
            now - timedelta(minutes=10), now - timedelta(minutes=5)
        )

        if previous:
            return self._compare_metric_dicts(recent, previous)

        # All traffic in one burst (typical for __main__): compare first vs second half of session
        queries = sorted(self.metrics["query_count"], key=lambda e: e.timestamp)
        if len(queries) < 4:
            return {}

        mid_time = queries[len(queries) // 2].timestamp
        older = self._calculate_windowed_metrics(queries[0].timestamp, mid_time)
        newer = self._calculate_windowed_metrics(mid_time, now)
        return self._compare_metric_dicts(newer, older)

    def _trigger_alert(
        self, metric: str, value: float, threshold: float, direction: str
    ):
        alert = {
            "timestamp": datetime.now(),
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "message": (
                f"Alert: {metric} is {value:.3f} ({direction} threshold {threshold})"
            ),
        }
        self.alerts.append(alert)
        self.logger.warning(alert["message"])

    def get_dashboard_metrics(self) -> dict[str, Any]:
        current_time = datetime.now()

        recent_metrics = self._calculate_windowed_metrics(
            current_time - timedelta(minutes=5), current_time
        )
        hourly_metrics = self._calculate_windowed_metrics(
            current_time - timedelta(hours=1), current_time
        )

        return {
            "hf_model": self.hf_model,
            "hf_provider": self.hf_provider,
            "embedding_model": HF_EMBEDDING_MODEL,
            "current": recent_metrics,
            "hourly": hourly_metrics,
            "trends": self._calculate_trends(),
            "alerts": self.alerts[-10:],
            "health_score": self._calculate_health_score(recent_metrics),
        }

    def _calculate_health_score(self, metrics: dict[str, float]) -> float:
        score = 100.0

        if metrics.get("error_rate", 0) > 0.01:
            score -= min(30, metrics["error_rate"] * 300)

        if metrics.get("p95_latency_ms", 0) > 2000:
            score -= min(20, (metrics["p95_latency_ms"] - 2000) / 100)

        if metrics.get("relevancy_score", 1) < 0.8:
            score -= min(30, (0.8 - metrics["relevancy_score"]) * 100)

        return max(0.0, score)


def _embedding_similarity(text_a: str, text_b: str) -> float:
    """Relevancy proxy via HF embeddings API."""
    api_key = (os.getenv("PRIMARY_LLM_KEY") or "").strip()
    client = InferenceClient(
        model=HF_EMBEDDING_MODEL,
        token=api_key,
        provider=HF_EMBEDDING_PROVIDER,
    )

    def encode(text: str) -> np.ndarray:
        resp = client.feature_extraction(text)
        vec = resp[0] if isinstance(resp[0], list) else resp
        return np.array(vec, dtype=float)

    a, b = encode(text_a), encode(text_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class RAGDashboard:
    """Simple monitoring dashboard"""

    def __init__(self, monitor: RAGMonitor):
        self.monitor = monitor

    def display_metrics(self):
        metrics = self.monitor.get_dashboard_metrics()

        print("=== RAG System Dashboard (Hugging Face) ===")
        print(f"Model: {metrics['hf_model']} (provider: {metrics['hf_provider']})")
        print(f"Embeddings: {metrics['embedding_model']} ({HF_EMBEDDING_PROVIDER})")
        print(f"Health Score: {metrics['health_score']:.1f}/100")
        print("\nCurrent Metrics (5 min):")
        for key, value in metrics["current"].items():
            print(f"  {key}: {value:.3f}")

        print("\nRecent Alerts:")
        for alert in metrics["alerts"][-5:]:
            ts = alert["timestamp"].strftime("%H:%M:%S")
            print(f"  [{ts}] {alert['message']}")

        print("\nTrends (newer half vs older half of session when run is short):")
        if not metrics["trends"]:
            print("  (not enough data yet)")
        for metric, trend in metrics["trends"].items():
            arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            print(f"  {metric}: {arrow} {abs(trend):.1%}")


if __name__ == "__main__":
    monitor = RAGMonitor()
    llm = get_hf_chat_llm()

    demo_queries = [
        {
            "question": "What is the capital of France?",
            "expected": "Paris is the capital of France.",
            "contexts": [
                "Paris is the capital and largest city of France.",
                "France is in Western Europe.",
            ],
        },
        {
            "question": "When was the Eiffel Tower completed?",
            "expected": "The Eiffel Tower was completed in 1889.",
            "contexts": [
                "The Eiffel Tower was completed in 1889.",
                "It is a landmark in Paris.",
            ],
        },
    ]

    print("Recording live HF queries...")
    for i, item in enumerate(demo_queries):
        monitor.track_hf_query(
            f"hf_{i}",
            item["question"],
            item["contexts"],
            llm,
            expected_answer=item["expected"],
        )

    # A few synthetic points to exercise alerts/dashboard (no extra HF calls)
    for i in range(10):
        monitor.track_query(
            f"sim_{i}",
            "Sample query",
            {"answer": "Sample response"},
            {
                "response_time": float(np.random.normal(1200, 300)),
                "relevancy_score": float(np.clip(np.random.normal(0.85, 0.1), 0, 1)),
                "chunks_retrieved": 3,
                "cost": float(np.random.normal(0.0004, 0.0001)),
                "error": "Timeout" if np.random.random() < 0.02 else None,
            },
        )

    RAGDashboard(monitor).display_metrics()
