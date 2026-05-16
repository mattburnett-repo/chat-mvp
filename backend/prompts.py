"""LLM and retrieval strings for the backend.

BGE embedding models (e.g. EMBEDDING_MODEL=bge-base-en-v1.5) were trained with one
instruction for search queries and plain text for indexed passages. Prepend
BGE_QUERY_PREFIX only in embed_query(), not when embedding document chunks.
"""

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Parenthesized string literals (not a """ text block) concatenate to one line with no extra newlines or indentation sent to the model.
PRIMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using only the provided context. "
    "If the context is insufficient, say so briefly. "
    'Use conversation history only to interpret follow-up questions (e.g. what "it" refers to); '
    "do not treat the history as factual sources. "
    "When relevant, mention which source URLs in the context support your answer."
)
