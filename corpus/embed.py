import os

from db import get_connection
from openai import OpenAI
from sql_queries import (
    SELECT_DOCUMENTS_WITHOUT_EMBEDDING,
    UPDATE_DOCUMENT_EMBEDDING,
)

client = OpenAI(api_key=os.environ["EMBEDDING_MODEL_KEY"])

# 1. fetch rows without embeddings
conn = get_connection()
cur = conn.cursor()

cur.execute(SELECT_DOCUMENTS_WITHOUT_EMBEDDING)

rows = cur.fetchall()

print(f"Found {len(rows)} rows to embed")

# 2. process each row
for doc_id, content in rows:
    print(f"Embedding id={doc_id}")

    response = client.embeddings.create(
        model=os.environ["EMBEDDING_MODEL"],
        input=content,
    )

    embedding = response.data[0].embedding

    cur.execute(UPDATE_DOCUMENT_EMBEDDING, (embedding, doc_id))

conn.commit()
cur.close()
conn.close()

print("Done")