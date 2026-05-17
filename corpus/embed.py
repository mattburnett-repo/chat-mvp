"""Offline batch job: fill documents.embedding for rows that are still NULL.

Handles the initial document embedding, after the documents are chunked and
saved to the database.

Run after corpus/ingest/crawl.py. Embedding calls live in embeddings.py so
ingest and /query use the same model and behavior. From repo root:
  .venv/bin/python -m corpus.embed
"""

from corpus.db import get_connection
from corpus.embeddings import embed_document
from corpus.sql_queries import (
    SELECT_DOCUMENTS_WITHOUT_EMBEDDING,
    UPDATE_DOCUMENT_EMBEDDING,
)


def main() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(SELECT_DOCUMENTS_WITHOUT_EMBEDDING)

    rows = cur.fetchall()

    print(f"Found {len(rows)} rows to embed")

    for doc_id, content in rows:
        print(f"Embedding id={doc_id}")
        embedding = embed_document(content)
        cur.execute(UPDATE_DOCUMENT_EMBEDDING, (embedding, doc_id))

    conn.commit()
    cur.close()
    conn.close()

    print("Done")


if __name__ == "__main__":
    main()
