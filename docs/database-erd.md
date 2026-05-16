# Database schema (Mermaid ER)

Tables and relationships as defined in `corpus/schema.py` (reference DDL) and used by `corpus/sql_queries.py` and the app.

## Entity relationship

```mermaid
erDiagram
  documents {
    int id PK
    text content
    vector embedding "pgvector; nullable until embed"
    text source_url
    int chunk_index
    text title
    timestamptz fetched_at
    jsonb metadata
  }
  conversations {
    text session_id PK
    timestamptz created_at
    text title
  }
  chat_messages {
    bigint id PK
    text session_id
    jsonb message "LangChain message dict"
    timestamptz created_at
  }
```

`conversations` and `chat_messages` are correlated by `session_id` at the application layer (`INSERT_CONVERSATION_IF_ABSENT` then chat rows for that id).

## Indexes and constraints (logical)

```mermaid
flowchart TB
  D[documents]
  D --> UQ["UNIQUE source_url + chunk_index"]
  D --> VEC["vector column for similarity ORDER BY embedding <-> query"]
  CM[chat_messages]
  CM --> IDX["INDEX session_id, created_at, id"]
  CV[conversations]
  CV --> PK["PK session_id"]
```

Vector dimension in schema reference: `vector(768)` for default `BAAI/bge-base-en-v1.5`.
