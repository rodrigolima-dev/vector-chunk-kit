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
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ValueError("embedding provider returned invalid indexes")
        return [list(item.embedding) for item in ordered]


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
        if len(items) != len(vectors) or not items:
            raise ValueError("chunks and vectors must have matching non-empty lengths")
        namespaces = {chunk.namespace for chunk in items}
        if len(namespaces) != 1 or not SAFE_ID.fullmatch(next(iter(namespaces))):
            raise ValueError("one valid namespace is required per batch")
        with self._connection.transaction():
            self._upsert_rows(items, vectors)

    def replace_source(
        self,
        namespace: str,
        source_id: str,
        current: list[Chunk],
        pending: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if not SAFE_ID.fullmatch(namespace) or not SAFE_ID.fullmatch(source_id):
            raise ValueError("namespace and source must be simple non-empty identifiers")
        if not current or any(
            chunk.namespace != namespace or chunk.source_id != source_id for chunk in current
        ):
            raise ValueError("replacement must contain one source")
        if sorted(chunk.chunk_index for chunk in current) != list(range(len(current))):
            raise ValueError("replacement chunk indexes must be contiguous")
        if len(pending) != len(vectors):
            raise ValueError("chunks and vectors must have matching lengths")
        expected = {chunk.chunk_id: chunk.content_hash for chunk in current}
        if any(expected.get(chunk.chunk_id) != chunk.content_hash for chunk in pending):
            raise ValueError("pending chunks must belong to the replacement")

        with self._connection.transaction():
            row = self._connection.execute(
                "SELECT 1 FROM chunkkit_namespace_models WHERE namespace = %s FOR UPDATE",
                (namespace,),
            ).fetchone()
            if row is None:
                raise ValueError("namespace lacks an embedding model record")
            if vectors:
                dimensions = len(vectors[0])
                if any(len(vector) != dimensions for vector in vectors):
                    raise ValueError("replacement vectors have different dimensions")
                self.ensure_dimensions(namespace, dimensions)

            rows = self._connection.execute(
                "SELECT chunk_id, content_hash FROM chunkkit_chunks "
                "WHERE namespace = %s AND source_id = %s AND chunk_id = ANY(%s)",
                (namespace, source_id, list(expected)),
            ).fetchall()
            existing: dict[str, str] = dict(rows)
            pending_ids = {chunk.chunk_id for chunk in pending}
            if any(
                existing.get(chunk_id) != content_hash
                for chunk_id, content_hash in expected.items()
                if chunk_id not in pending_ids
            ):
                raise ValueError("source changed during embedding; retry replacement")

            self._upsert_rows(pending, vectors)
            self._connection.execute(
                "DELETE FROM chunkkit_chunks "
                "WHERE namespace = %s AND source_id = %s AND NOT (chunk_id = ANY(%s))",
                (namespace, source_id, list(expected)),
            )

    def _upsert_rows(self, items: list[Chunk], vectors: list[list[float]]) -> None:
        from pgvector import Vector
        from psycopg.types.json import Jsonb

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
