"""Optional external adapters. Importing this module does not make network calls."""

from typing import Self

from .prepare import SAFE_ID, Chunk


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        if not api_key or not model:
            raise ValueError("embedding credentials and model are required")
        self._client = OpenAI(api_key=api_key, max_retries=2, timeout=30.0)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding input must contain non-empty text")
        response = self._client.embeddings.create(
            model=self._model, input=texts, encoding_format="float"
        )
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]


class PgvectorStore:
    """Store only in the fixed chunkkit_chunks table supplied by schema.sql."""

    def __init__(self, database_url: str) -> None:
        from pgvector.psycopg import register_vector
        from psycopg import connect

        if not database_url:
            raise ValueError("database URL is required")
        self._connection = connect(database_url, autocommit=True, connect_timeout=10)
        try:
            register_vector(self._connection)
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self._connection.close()

    def ensure_model(self, namespace: str, model: str) -> None:
        if not SAFE_ID.fullmatch(namespace):
            raise ValueError("namespace must be a simple non-empty identifier")
        if not model.strip() or len(model) > 128:
            raise ValueError("embedding model must be a non-empty identifier")
        with self._connection.transaction():
            row = self._connection.execute(
                "INSERT INTO chunkkit_namespace_models (namespace, embedding_model) "
                "SELECT %s, %s WHERE NOT EXISTS "
                "(SELECT 1 FROM chunkkit_chunks WHERE namespace = %s) "
                "ON CONFLICT (namespace) DO NOTHING "
                "RETURNING embedding_model",
                (namespace, model, namespace),
            ).fetchone()
            if row is None:
                row = self._connection.execute(
                    "SELECT embedding_model FROM chunkkit_namespace_models WHERE namespace = %s",
                    (namespace,),
                ).fetchone()
            if row is None:
                raise ValueError("legacy namespace has chunks without a model record")
            if row[0] != model:
                raise ValueError("namespace uses a different embedding model")

    def ensure_dimensions(self, namespace: str, dimensions: int) -> None:
        if not SAFE_ID.fullmatch(namespace):
            raise ValueError("namespace must be a simple non-empty identifier")
        if not 1 <= dimensions <= 4096:
            raise ValueError("vector dimension must be between 1 and 4096")
        row = self._connection.execute(
            "UPDATE chunkkit_namespace_models SET dimensions = %s "
            "WHERE namespace = %s AND (dimensions IS NULL OR dimensions = %s) "
            "RETURNING dimensions",
            (dimensions, namespace, dimensions),
        ).fetchone()
        if row is None:
            raise ValueError("namespace uses a different vector dimension or lacks a model record")

    def existing_hashes(self, namespace: str, ids: list[str]) -> dict[str, str]:
        if not SAFE_ID.fullmatch(namespace):
            raise ValueError("namespace must be a simple non-empty identifier")
        if not ids:
            return {}
        rows = self._connection.execute(
            "SELECT chunk_id, content_hash FROM chunkkit_chunks "
            "WHERE namespace = %s AND chunk_id = ANY(%s)",
            (namespace, ids),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def upsert(self, items: list[Chunk], vectors: list[list[float]]) -> None:
        from pgvector import Vector
        from psycopg.types.json import Jsonb

        if len(items) != len(vectors) or not items:
            raise ValueError("chunks and vectors must have matching non-empty lengths")
        namespaces = {chunk.namespace for chunk in items}
        if len(namespaces) != 1 or not SAFE_ID.fullmatch(next(iter(namespaces))):
            raise ValueError("one valid namespace is required per batch")
        statement = (
            "INSERT INTO chunkkit_chunks "
            "(namespace, source_id, chunk_index, chunk_id, content_hash, "
            "content, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (namespace, source_id, chunk_index) DO UPDATE SET "
            "chunk_id = EXCLUDED.chunk_id, content_hash = EXCLUDED.content_hash, "
            "content = EXCLUDED.content, metadata = EXCLUDED.metadata, "
            "embedding = EXCLUDED.embedding, updated_at = now()"
        )
        with self._connection.transaction():
            for chunk, vector in zip(items, vectors, strict=True):
                self._connection.execute(
                    statement,
                    (
                        chunk.namespace,
                        chunk.source_id,
                        chunk.chunk_index,
                        chunk.chunk_id,
                        chunk.content_hash,
                        chunk.text,
                        Jsonb(chunk.metadata),
                        Vector(vector),
                    ),
                )
