import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

_backend_dir = Path(__file__).resolve().parent
_repo_root = _backend_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from utils.env_loader import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

from .retrieve import search_by_embedding  # noqa: E402


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(int(os.environ["QUERY_TOP_K"]), ge=1, le=20)


class SourceRef(BaseModel):
    source_url: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


app = FastAPI(title="Chat MVP API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    emb_model = os.environ.get("OPENAI_EMBEDDING_MODEL")
    if not emb_model:
        raise HTTPException(status_code=500, detail="Missing OPENAI_EMBEDDING_MODEL")

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
        return QueryResponse(
            answer="No matching documents were found in the database.",
            sources=[],
        )

    context_blocks = []
    sources: list[SourceRef] = []
    for source_url, chunk_index, content in rows:
        sources.append(SourceRef(source_url=source_url, chunk_index=chunk_index))
        context_blocks.append(
            f"[source_url={source_url!r} chunk_index={chunk_index}]\n{content}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    llm = ChatOpenAI(model=os.environ["OPENAI_CHAT_MODEL"], temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer using only the provided context. "
                "If the context is insufficient, say so briefly. "
                "When relevant, mention which source URLs support your answer.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke({"context": context, "question": req.query})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {e}") from e

    return QueryResponse(answer=answer, sources=sources)
