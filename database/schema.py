"""Reference DDL for the project database (documentation only; not executed by imports).

Copy into pgAdmin or psql as needed. Requires pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`
"""

# Full schema as exported / maintained for the `documents` table and indexes.
SCHEMA_SQL = r"""
-- Table: public.documents

-- DROP TABLE IF EXISTS public.documents;

CREATE TABLE IF NOT EXISTS public.documents
(
    id integer NOT NULL DEFAULT nextval('documents_id_seq'::regclass),
    content text COLLATE pg_catalog."default",
    embedding vector(1536),
    source_url text COLLATE pg_catalog."default" NOT NULL,
    chunk_index integer NOT NULL DEFAULT 0,
    title text COLLATE pg_catalog."default",
    fetched_at timestamp with time zone,
    metadata jsonb DEFAULT '{}'::jsonb,
    CONSTRAINT documents_pkey PRIMARY KEY (id)
)
TABLESPACE pg_default;

-- Owner varies by environment (e.g. local vs Render).
-- ALTER TABLE IF EXISTS public.documents OWNER TO your_db_user;

-- Index: documents_source_chunk_idx

-- DROP INDEX IF EXISTS public.documents_source_chunk_idx;

CREATE UNIQUE INDEX IF NOT EXISTS documents_source_chunk_idx
    ON public.documents USING btree
    (source_url COLLATE pg_catalog."default" ASC NULLS LAST, chunk_index ASC NULLS LAST)
    TABLESPACE pg_default;
"""
