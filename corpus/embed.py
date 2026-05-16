"""Offline batch job: fill documents.embedding for rows that are still NULL.

Handles the initial document embedding, after the documents are chunked and saved to the database.

Run after corpus/ingest/crawl.py. Embedding calls live in embeddings.py so
ingest and /query use the same model and behavior. From repo root:
  cd corpus && ../.venv/bin/python embed.py
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_corpus_dir = Path(__file__).resolve().parent
for _p in (_repo_root, _corpus_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.env_loader import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

from db import get_connection  # noqa: E402
from embeddings import embed_document  # noqa: E402
from sql_queries import (  # noqa: E402
    SELECT_DOCUMENTS_WITHOUT_EMBEDDING,
    UPDATE_DOCUMENT_EMBEDDING,
)

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
