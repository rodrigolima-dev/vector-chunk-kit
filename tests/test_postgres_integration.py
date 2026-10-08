"""Exercise the SQL adapter against a disposable, local PostgreSQL instance."""

import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from vector_chunk_kit.adapters import PgvectorStore
from vector_chunk_kit.ingest import ingest_chunks
from vector_chunk_kit.prepare import prepare_documents


class FakeEmbedder:
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0] * self.dimensions for _ in texts]


def test_pgvector_round_trip_and_guards() -> None:
    database_url = os.environ.get("CHUNKKIT_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("disposable local PostgreSQL is not configured")
    connection_info = conninfo_to_dict(database_url)
    if connection_info.get("host") not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("integration test refuses a non-local database")
    if connection_info.get("dbname") != "chunkkit_test":
        pytest.fail("integration test requires the disposable chunkkit_test database")

    schema_name = f"chunkkit_test_{uuid4().hex}"
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        try:
            scoped_url = make_conninfo(
                database_url, options=f"-csearch_path={schema_name},public"
            )
            with psycopg.connect(scoped_url, autocommit=True) as setup:
                schema_sql = Path("schema.sql").read_text(encoding="utf-8")
                for statement in schema_sql.split(";"):
                    if statement.strip():
                        setup.execute(statement)

            def prepare(body: str):
                return prepare_documents(
                    [{"id": "synthetic", "namespace": "alpha", "title": "Example", "text": body}],
                    "alpha",
                    200,
                    0,
                    "{body}",
                )

            first = prepare("First synthetic paragraph")
            second = prepare("Changed synthetic paragraph")
            assert first[0].chunk_id == second[0].chunk_id
            embedder = FakeEmbedder()
            with PgvectorStore(scoped_url) as store:
                first_run = ingest_chunks(first, "alpha", embedder, store, 1, "offline-model")
                assert first_run.written == 1
                repeated_run = ingest_chunks(first, "alpha", embedder, store, 1, "offline-model")
                assert repeated_run.skipped == 1
                assert embedder.calls == 1
                changed_run = ingest_chunks(second, "alpha", embedder, store, 1, "offline-model")
                assert changed_run.written == 1
                assert store.existing_hashes("alpha", [second[0].chunk_id]) == {
                    second[0].chunk_id: second[0].content_hash
                }
                assert store.existing_hashes("beta", [second[0].chunk_id]) == {}

                with pytest.raises(ValueError, match="different embedding model"):
                    ingest_chunks(second, "alpha", embedder, store, 1, "other-model")
                with pytest.raises(ValueError, match="different vector dimension"):
                    ingest_chunks(first, "alpha", FakeEmbedder(2), store, 1, "offline-model")

            with psycopg.connect(scoped_url, autocommit=True) as verify:
                row = verify.execute(
                    "SELECT content_hash, embedding::text FROM chunkkit_chunks "
                    "WHERE namespace = %s AND source_id = %s",
                    ("alpha", "synthetic"),
                ).fetchone()
                assert row is not None
                assert row[0] == second[0].content_hash
                assert row[1] == "[1,1,1]"
        finally:
            admin.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name))
            )
