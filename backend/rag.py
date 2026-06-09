"""Callable RAG pipeline shared by the API and evaluation harness."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from corpus.embeddings import embed_query
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langsmith import traceable

from backend.prompts import PRIMARY_SYSTEM_PROMPT
from backend.retrieve import search_by_embedding

NO_MATCH_ANSWER = "No matching documents were found in the database."


@dataclass(frozen=True)
class SourceRef:
    source_url: str
    chunk_index: int


@dataclass(frozen=True)
class RAGQueryResult:
    question: str
    answer: str
    sources: list[SourceRef]
    retrieved_contexts: list[str]
    latency_ms: float


def default_top_k() -> int:
    return int(os.environ["QUERY_TOP_K"])


def format_retrieval(
    rows: list[tuple[str, int, str]],
) -> tuple[str, list[SourceRef], list[str]]:
    sources: list[SourceRef] = []
    context_blocks: list[str] = []
    retrieved_contexts: list[str] = []
    for source_url, chunk_index, content in rows:
        sources.append(SourceRef(source_url=source_url, chunk_index=chunk_index))
        block = f"[source_url={source_url!r} chunk_index={chunk_index}]\n{content}"
        context_blocks.append(block)
        retrieved_contexts.append(content)
    context = "\n\n---\n\n".join(context_blocks)
    return context, sources, retrieved_contexts


_llm: ChatHuggingFace | None = None
_chain = None


def _get_llm() -> ChatHuggingFace:
    global _llm
    if _llm is None:
        _llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                model=os.environ["PRIMARY_LLM_MODEL"],
                task="text-generation",
                provider=os.environ["PRIMARY_LLM_PROVIDER"],
                temperature=0,
                do_sample=False,
                huggingfacehub_api_token=os.environ["PRIMARY_LLM_KEY"],
            )
        )
    return _llm


def _get_chain():
    global _chain
    if _chain is None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PRIMARY_SYSTEM_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )
        _chain = prompt | _get_llm() | StrOutputParser()
    return _chain


def generate_answer(question: str, context: str) -> str:
    return _get_chain().invoke({"context": context, "question": question})


@traceable(name="rag_query", run_type="chain")
def run_rag_query(
    question: str,
    top_k: int | None = None,
    *,
    generate: bool = True,
) -> RAGQueryResult:
    """Embed, retrieve from pgvector, optionally generate an answer."""
    start = time.perf_counter()
    k = top_k if top_k is not None else default_top_k()
    q_vec = embed_query(question)
    rows = search_by_embedding(q_vec, k)

    if not rows:
        elapsed = (time.perf_counter() - start) * 1000
        return RAGQueryResult(
            question=question,
            answer=NO_MATCH_ANSWER if generate else "",
            sources=[],
            retrieved_contexts=[],
            latency_ms=elapsed,
        )

    context, sources, retrieved_contexts = format_retrieval(rows)
    answer = generate_answer(question, context) if generate else ""
    elapsed = (time.perf_counter() - start) * 1000
    return RAGQueryResult(
        question=question,
        answer=answer,
        sources=sources,
        retrieved_contexts=retrieved_contexts,
        latency_ms=elapsed,
    )
