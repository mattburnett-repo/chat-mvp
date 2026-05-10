"""Reference DDL for the project database (documentation only; not executed by imports).

Copy into pgAdmin or psql as needed. Requires pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`
"""

# Option A — session-scoped chat memory (LangChain-friendly JSONB per message).
# Optional `conversations` row is not required for storage but supports titles / last-active later.
SCHEMA_CONVERSATIONS_SQL = r"""
-- Table: public.conversations (session metadata; optional for MVP)

-- DROP TABLE IF EXISTS public.conversations;

CREATE TABLE IF NOT EXISTS public.conversations
(
    session_id text COLLATE pg_catalog."default" NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    title text COLLATE pg_catalog."default",
    CONSTRAINT conversations_pkey PRIMARY KEY (session_id)
)
TABLESPACE pg_default;
"""

SCHEMA_CHAT_MESSAGES_SQL = r"""
-- Table: public.chat_messages (one LangChain BaseMessage per row, JSONB)

-- DROP TABLE IF EXISTS public.chat_messages;

CREATE TABLE IF NOT EXISTS public.chat_messages
(
    id bigserial NOT NULL,
    session_id text COLLATE pg_catalog."default" NOT NULL,
    message jsonb NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT chat_messages_pkey PRIMARY KEY (id)
)
TABLESPACE pg_default;

-- Index: chat_messages_session_created_idx

-- DROP INDEX IF EXISTS public.chat_messages_session_created_idx;

CREATE INDEX IF NOT EXISTS chat_messages_session_created_idx
    ON public.chat_messages USING btree
    (session_id COLLATE pg_catalog."default" ASC NULLS LAST, created_at ASC NULLS LAST, id ASC NULLS LAST)
    TABLESPACE pg_default;
"""

# Full schema as exported / maintained for the `documents` table and indexes.
SCHEMA_DOCUMENTS_SQL = r"""
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
