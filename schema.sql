-- Apply only to a database you own. This file is never executed by the CLI.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunkkit_namespace_models (
    namespace text PRIMARY KEY,
    embedding_model text NOT NULL
);

CREATE TABLE IF NOT EXISTS chunkkit_chunks (
    chunk_id text PRIMARY KEY,
    namespace text NOT NULL REFERENCES chunkkit_namespace_models(namespace),
    source_id text NOT NULL,
    chunk_index integer NOT NULL CHECK (chunk_index >= 0),
    content_hash text NOT NULL,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (namespace, source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunkkit_chunks_namespace_source_idx
    ON chunkkit_chunks (namespace, source_id);
