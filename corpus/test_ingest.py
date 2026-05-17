from corpus.db import get_connection
from corpus.embeddings import embed_query
from corpus.sql_queries import SELECT_DOCUMENTS_BY_VECTOR_SIMILARITY


def _to_pgvector_literal(vec: list[float]) -> str:
    """psycopg2 sends Python lists as numeric[]; pgvector needs a vector cast."""
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def search(query, top_k=5):
    q_emb = embed_query(query)

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
