# Ingestion pipeline (Mermaid)

Offline jobs: `corpus/ingest/crawl.py` writes chunked rows to `documents`; `corpus/embed.py` fills `embedding` for rows where it is null.

## Crawl pipeline

```mermaid
flowchart TD
  START[crawl.py main] --> GH[iter_github_readme_docs]
  GH --> API[GitHub REST: repo + readme metadata]
  API --> RAW[README text via download_url]
  RAW --> LINKS[Parse markdown links one hop]
  LINKS --> FILTER[_dev_file_path_allowed]
  FILTER --> RGET[GET raw.githubusercontent.com per path]
  START --> SEEDS[Iterate sources.SEED_URLS]
  SEEDS --> CANON[canonical_url dedupe]
  CANON --> HTTP[httpx GET HTML only]
  HTTP --> TRAF[trafilatura extract text]
  TRAF --> BS[BeautifulSoup title + strip tags for chunking]
  GH --> STORE[_store_chunked_text]
  BS --> STORE
  STORE --> DEL[DELETE documents for source_url]
  DEL --> INS[INSERT chunks with metadata jsonb]
  INS --> COMMIT[conn.commit per successful source]
```

## GitHub README branch (conceptual)

```mermaid
flowchart LR
  subgraph github_readme.py
    R[GET api.github.com/repos/o/r]
    RM[GET .../readme]
    DL[GET download_url text]
    HOP[Resolve blob/raw links to repo paths]
    FETCH[GET raw content for each allowed path]
  end
  R --> RM --> DL --> HOP --> FETCH
```

Constants `GITHUB_OWNER` / `GITHUB_REPO` and `SEED_URLS` live in `corpus/ingest/sources.py`.

## Chunking

```mermaid
flowchart LR
  T[Plain text] --> CH[chunking.chunk_text]
  CH --> P["Windows: CHUNK_MAX_CHARS overlap CHUNK_OVERLAP_CHARS"]
  P --> L[List of non-empty strings]
```

## Embed job

```mermaid
flowchart TD
  E[embed.py from corpus/ cwd] --> SEL[SELECT id, content WHERE embedding IS NULL]
  SEL --> LOOP{For each row}
  LOOP --> OAI[OpenAI embeddings.create]
  OAI --> UPD[UPDATE documents SET embedding]
  UPD --> LOOP
  LOOP --> DONE[commit close]
```

## Env knobs (ingest)

| Area | Variables (from code / README) |
|------|----------------------------------|
| Crawl | `CRAWL_MAX_PAGES`, `CRAWL_REQUEST_TIMEOUT_S`, `CRAWL_DELAY_S`, `CRAWL_USER_AGENT` |
| Chunks | `CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_CHARS` |
| DB / OpenAI | Same as app: `PG*`, `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL` |
