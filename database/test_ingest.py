import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_database_dir = Path(__file__).resolve().parent
for _p in (_repo_root, _database_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.env_loader import load_repo_dotenv #noqa E402
from openai import OpenAI #noqa E402

load_repo_dotenv()

from db import get_connection  # noqa: E402
from sql_queries import SELECT_DOCUMENTS_BY_VECTOR_SIMILARITY  # noqa: E402

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed(text: str):
    resp = client.embeddings.create(
        model=os.environ["OPENAI_EMBEDDING_MODEL"],
        input=text,
    )
    return resp.data[0].embedding


def _to_pgvector_literal(vec: list[float]) -> str:
    """psycopg2 sends Python lists as numeric[]; pgvector needs a vector cast."""
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def search(query, top_k=5):
    q_emb = embed(query)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        SELECT_DOCUMENTS_BY_VECTOR_SIMILARITY,
        (_to_pgvector_literal(q_emb), top_k),
    )

    results = cur.fetchall()
    cur.close()
    conn.close()
    return results


if __name__ == "__main__":
    query = "what is this system about?"

    results = search(query)

    print("\nTOP MATCHES:\n")
    for r in results:
        print("URL:", r[0])
        print("Chunk:", r[1])
        print("Text:", r[2][:300])
        print("-" * 80)
