# Architecture overview (Mermaid)

High-level layout of the Chat MVP repo: Streamlit UI, FastAPI backend, PostgreSQL with pgvector, Hugging Face (chat + local embeddings), and offline ingestion scripts.

## System context

```mermaid
flowchart LR
  subgraph Clients
    UI[Streamlit frontend/app.py]
  end
  subgraph Backend
    API[FastAPI corpus/main.py]
    RET[retrieve.search_by_embedding]
  end
  subgraph Data
    PG[(PostgreSQL + pgvector)]
  end
  subgraph External
    HF[Hugging Face Inference API]
  end
  UI -->|HTTP POST /query| API
  API --> RET
  RET --> PG
  API -->|embed_query + chat| HF
```

## Repository modules

```mermaid
flowchart TB
  subgraph frontend
    APP[app.py]
  end
  subgraph corpus
    MAIN[main.py]
    RET[retrieve.py]
  end
  subgraph corpus
    DB[db.py]
    SCH[schema.py DDL reference]
    SQL[sql_queries.py]
    PCH[postgres_chat_message_history.py]
    subgraph ingest
      CRAWL[crawl.py]
      GH[github_readme.py]
      SRC[sources.py]
      CHK[chunking.py]
    end
    EMBED[embed.py]
  end
  subgraph utils
    ENV[env_loader.py]
  end
  APP --> ENV
  MAIN --> ENV
  MAIN --> DB
  MAIN --> SQL
  MAIN --> PCH
  MAIN --> RET
  RET --> DB
  RET --> SQL
  PCH --> DB
  PCH --> SQL
  CRAWL --> ENV
  CRAWL --> DB
  CRAWL --> SQL
  CRAWL --> GH
  CRAWL --> SRC
  CRAWL --> CHK
  EMBED --> DB
  EMBED --> SQL
```

## Runtime processes

```mermaid
flowchart LR
  subgraph Dev
    U[uvicorn corpus.main:app]
    S[streamlit run frontend/app.py]
  end
  subgraph Batch
    C[python corpus/ingest/crawl.py]
    E[cd corpus && python embed.py]
  end
  S -->|CHAT_MVP_API_BASE_URL| U
  C --> PG[(PostgreSQL)]
  E --> PG
  U --> PG
```

`embed.py` is intended to be run from the `corpus/` working directory (its imports use `db` and `sql_queries` as top-level modules).
