import os
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

_backend_dir = Path(__file__).resolve().parent
_repo_root = _backend_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from utils.env_loader import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

from database.db import get_connection  # noqa: E402
from database.postgres_chat_message_history import PostgresChatMessageHistory  # noqa: E402
from database.sql_queries import INSERT_CONVERSATION_IF_ABSENT  # noqa: E402
from .retrieve import search_by_embedding  # noqa: E402


def get_session_history(session_id: str) -> PostgresChatMessageHistory:
    max_toks = int(os.environ.get("CHAT_HISTORY_MAX_TOKENS", "3000"))
    return PostgresChatMessageHistory(session_id=session_id, max_history_tokens=max_toks)


def _ensure_conversation_row(session_id: str) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(INSERT_CONVERSATION_IF_ABSENT, (session_id,))
        conn.commit()
        cur.close()
    finally:
        conn.close()


_llm = ChatOpenAI(model=os.environ["OPENAI_CHAT_MODEL"], temperature=0)

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer using only the provided context. "
            "If the context is insufficient, say so briefly. "
            "Use conversation history only to interpret follow-up questions (e.g. what \"it\" refers to); "
            "do not treat the history as factual sources. "
            "When relevant, mention which source URLs in the context support your answer.",
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
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    emb_model = os.environ.get("OPENAI_EMBEDDING_MODEL")
    if not emb_model:
        raise HTTPException(status_code=500, detail="Missing OPENAI_EMBEDDING_MODEL")

    session_id = _effective_session_id(req.session_id)
    try:
        _ensure_conversation_row(session_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database error: {e}") from e

    embeddings = OpenAIEmbeddings(model=emb_model)
    try:
        q_vec = embeddings.embed_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}") from e

    try:
        rows = search_by_embedding(q_vec, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database query failed: {e}") from e

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

    try:
        answer = _chain_with_history.invoke(
            {"context": context, "question": req.query},
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {e}") from e

    return QueryResponse(answer=answer, sources=sources, session_id=session_id)
