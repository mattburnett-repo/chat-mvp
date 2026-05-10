# RAG query and chat flow (Mermaid)

End-to-end behavior of `POST /query` in `backend/main.py`, the vector retrieval layer, and how the Streamlit client participates.

## Sequence: happy path

```mermaid
sequenceDiagram
  participant ST as Streamlit app.py
  participant API as FastAPI /query
  participant OAI as OpenAI embeddings + chat
  participant PG as PostgreSQL
  participant H as PostgresChatMessageHistory

  ST->>API: POST JSON query, top_k, session_id?
  API->>PG: INSERT conversation if absent
  API->>OAI: embed_query(query)
  OAI-->>API: query vector
  API->>PG: SELECT top_k by embedding distance
  PG-->>API: rows source_url, chunk_index, content
  API->>H: RunnableWithMessageHistory loads trimmed history
  API->>OAI: ChatOpenAI with context + question + history
  OAI-->>API: answer string
  API-->>ST: answer, sources[], session_id
```

## Sequence: no matching documents

```mermaid
sequenceDiagram
  participant API as FastAPI /query
  participant PG as PostgreSQL
  participant H as PostgresChatMessageHistory

  API->>PG: vector search
  PG-->>API: zero rows
  API->>H: add_messages Human + AI canned reply
  API-->>API: QueryResponse canned, sources empty
```

## Backend decision flow

```mermaid
flowchart TD
  A[POST /query] --> B{OPENAI_API_KEY and embedding model set?}
  B -->|no| E500[HTTP 500]
  B -->|yes| C[Resolve session_id UUID or client id]
  C --> D[Ensure conversations row]
  D --> F[Embed user query]
  F --> G[search_by_embedding top_k]
  G --> H{Any rows?}
  H -->|no| I[Save Human+AI canned to history]
  I --> J[Return answer + empty sources]
  H -->|yes| K[Build context blocks from rows]
  K --> L[RunnableWithMessageHistory invoke]
  L --> M[Return answer + SourceRef list]
```

## Streamlit session state

```mermaid
stateDiagram-v2
  [*] --> Empty: first load
  Empty --> Chatting: user sends message
  Chatting --> Chatting: query_api returns session_id stored in conversation_id
  Chatting --> Empty: sidebar New chat clears messages and conversation_id
```

Environment: `CHAT_MVP_API_BASE_URL` (default `http://127.0.0.1:8000`), `QUERY_TOP_K` default for UI and server default on `QueryRequest`.
