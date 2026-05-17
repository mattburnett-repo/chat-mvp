import os
import time
import uuid

from corpus.db import get_connection
from corpus.embeddings import embed_query
from corpus.postgres_chat_message_history import PostgresChatMessageHistory
from corpus.sql_queries import INSERT_CONVERSATION_IF_ABSENT
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

from .prompts import PRIMARY_SYSTEM_PROMPT
from .retrieve import search_by_embedding


def get_session_history(session_id: str) -> PostgresChatMessageHistory:
    max_toks = int(os.environ["CHAT_HISTORY_MAX_TOKENS"])
    return PostgresChatMessageHistory(
        session_id=session_id, max_history_tokens=max_toks
    )


def _ensure_conversation_row(session_id: str) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(INSERT_CONVERSATION_IF_ABSENT, (session_id,))
        conn.commit()
        cur.close()
    finally:
        conn.close()


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

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            PRIMARY_SYSTEM_PROMPT,
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

_chain_with_history = RunnableWithMessageHistory(
    _RAG_PROMPT | _llm | StrOutputParser(),
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(int(os.environ["QUERY_TOP_K"]), ge=1, le=20)
    session_id: str | None = Field(default=None, max_length=256)


class SourceRef(BaseModel):
    source_url: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    session_id: str


app = FastAPI(title="Chat MVP API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


def _effective_session_id(raw: str | None) -> str:
    if raw and raw.strip():
        return raw.strip()
    return str(uuid.uuid4())


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not os.environ.get("PRIMARY_LLM_KEY"):
        raise HTTPException(status_code=500, detail="Missing PRIMARY_LLM_KEY")
    session_id = _effective_session_id(req.session_id)
    try:
        _ensure_conversation_row(session_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database error: {e}") from e

    try:
        q_vec = embed_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}") from e

    try:
        rows = search_by_embedding(q_vec, req.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Database query failed: {e}"
        ) from e

    if not rows:
        canned = "No matching documents were found in the database."
        try:
            get_session_history(session_id).add_messages(
                [HumanMessage(content=req.query), AIMessage(content=canned)]
            )
        except Exception as e:
            raise HTTPException(
                status_code=502, detail=f"Failed to save chat history: {e}"
            ) from e
        return QueryResponse(answer=canned, sources=[], session_id=session_id)

    context_blocks = []
    sources: list[SourceRef] = []
    for source_url, chunk_index, content in rows:
        sources.append(SourceRef(source_url=source_url, chunk_index=chunk_index))
        context_blocks.append(
            f"[source_url={source_url!r} chunk_index={chunk_index}]\n{content}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    for attempt in range(4):
        try:
            answer = _chain_with_history.invoke(
                {"context": context, "question": req.query},
                config={"configurable": {"session_id": session_id}},
            )
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            if attempt < 3 and "429" in str(e).lower():
                time.sleep(2)
                continue
            raise HTTPException(status_code=502, detail=f"LLM failed: {e}") from e

    raise HTTPException(status_code=502, detail="LLM failed")
