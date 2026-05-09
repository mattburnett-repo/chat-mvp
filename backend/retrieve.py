"""Vector search against the shared `database` layer."""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from database.db import get_connection  # noqa: E402
from database.sql_queries import SELECT_DOCUMENTS_BY_VECTOR_SIMILARITY  # noqa: E402
from utils.env_loader import load_repo_dotenv  # noqa: E402

load_repo_dotenv()


def _to_pgvector_literal(vec: list[float]) -> str:
    """psycopg2 sends Python lists as numeric[]; pgvector needs a vector cast."""
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def search_by_embedding(embedding: list[float], top_k: int) -> list[tuple[str, int, str]]:
    """Return rows: (source_url, chunk_index, content)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            SELECT_DOCUMENTS_BY_VECTOR_SIMILARITY,
            (_to_pgvector_literal(embedding), top_k),
        )
        rows = cur.fetchall()
        cur.close()
        return list(rows)
    finally:
        conn.close()
