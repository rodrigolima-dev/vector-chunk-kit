from contextlib import nullcontext
from types import SimpleNamespace

import openai
import pgvector.psycopg
import psycopg
import pytest

from vector_chunk_kit.adapters import OpenAIEmbedder, PgvectorStore
from vector_chunk_kit.prepare import prepare_documents


class FakeConnection:
    def __init__(self) -> None:
        self.queries: list[tuple[str, object]] = []
        self.closed = False
        self.transactions = 0

    def execute(self, statement: str, params: object) -> "FakeConnection":
        self.queries.append((statement, params))
        return self

    def fetchall(self) -> list[tuple[str, str]]:
        return [("existing-id", "existing-hash")]

    def transaction(self):
        self.transactions += 1
        return nullcontext()

    def close(self) -> None:
        self.closed = True


def make_chunk(namespace: str):
    return prepare_documents(
        [{"id": "source", "namespace": namespace, "title": "Synthetic", "text": "safe text"}],
        namespace,
        20,
        0,
        "{body}",
    )[0]


def test_pgvector_adapter_rejects_mixed_namespace_batch(monkeypatch) -> None:
    connection = FakeConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)

    with PgvectorStore("dbname=synthetic") as store:
        with pytest.raises(ValueError, match="namespace"):
            store.upsert([make_chunk("one"), make_chunk("two")], [[1.0], [1.0]])

    assert connection.queries == []
    assert connection.closed


def test_pgvector_read_keeps_namespace_in_parameterized_query(monkeypatch) -> None:
    connection = FakeConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)

    with PgvectorStore("dbname=synthetic") as store:
        assert store.existing_hashes("demo", ["existing-id"]) == {"existing-id": "existing-hash"}

    statement, params = connection.queries[0]
    assert "WHERE namespace = %s" in statement
    assert "demo" not in statement
    assert params == ("demo", ["existing-id"])


def test_pgvector_model_guard_rejects_a_different_model(monkeypatch) -> None:
    class ModelConnection(FakeConnection):
        model: str | None = None
        row: tuple[str] | None = None

        def execute(self, statement: str, params: object) -> "ModelConnection":
            super().execute(statement, params)
            if statement.startswith("INSERT INTO chunkkit_namespace_models"):
                assert isinstance(params, tuple)
                requested_model = params[1]
                assert isinstance(requested_model, str)
                self.row = (requested_model,) if self.model is None else None
                self.model = self.model or requested_model
            elif statement.startswith("SELECT embedding_model FROM chunkkit_namespace_models"):
                self.row = (self.model,) if self.model is not None else None
            return self

        def fetchone(self) -> tuple[str] | None:
            return self.row

    connection = ModelConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)

    with PgvectorStore("dbname=synthetic") as store:
        store.ensure_model("demo", "first-model")
        store.ensure_model("demo", "first-model")
        with pytest.raises(ValueError, match="different embedding model"):
            store.ensure_model("demo", "second-model")

    assert connection.model == "first-model"
    assert all("demo" not in statement for statement, _ in connection.queries)
    assert connection.closed


def test_pgvector_refuses_legacy_chunks_without_a_model_record(monkeypatch) -> None:
    class LegacyConnection(FakeConnection):
        row: tuple[str] | None = None

        def execute(self, statement: str, params: object) -> "LegacyConnection":
            super().execute(statement, params)
            if statement.startswith("INSERT INTO chunkkit_namespace_models"):
                assert isinstance(params, tuple)
                self.row = None if "NOT EXISTS" in statement else (params[1],)
            elif statement.startswith("SELECT embedding_model FROM chunkkit_namespace_models"):
                self.row = None
            return self

        def fetchone(self) -> tuple[str] | None:
            return self.row

    connection = LegacyConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)

    with PgvectorStore("dbname=synthetic") as store:
        with pytest.raises(ValueError, match="legacy namespace"):
            store.ensure_model("demo", "new-model")

    assert connection.closed


def test_pgvector_dimension_guard_rejects_a_different_size(monkeypatch) -> None:
    class DimensionConnection(FakeConnection):
        dimensions: int | None = None
        row: tuple[int] | None = None

        def execute(self, statement: str, params: object) -> "DimensionConnection":
            super().execute(statement, params)
            if statement.startswith("UPDATE chunkkit_namespace_models"):
                assert isinstance(params, tuple)
                requested = params[0]
                assert isinstance(requested, int)
                if self.dimensions is None or self.dimensions == requested:
                    self.dimensions = requested
                    self.row = (requested,)
                else:
                    self.row = None
            return self

        def fetchone(self) -> tuple[int] | None:
            return self.row

    connection = DimensionConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)

    with PgvectorStore("dbname=synthetic") as store:
        store.ensure_dimensions("demo", 2)
        store.ensure_dimensions("demo", 2)
        with pytest.raises(ValueError, match="different vector dimension"):
            store.ensure_dimensions("demo", 3)

    assert connection.dimensions == 2
    assert all("demo" not in statement for statement, _ in connection.queries)
    assert connection.closed


def test_pgvector_replaces_one_source_in_one_transaction(monkeypatch) -> None:
    class ReplacementConnection(FakeConnection):
        def fetchone(self) -> tuple[int]:
            return (1,)

        def fetchall(self) -> list[tuple[str, str]]:
            return []

    connection = ReplacementConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)
    chunk = make_chunk("demo")

    with PgvectorStore("dbname=synthetic") as store:
        store.replace_source("demo", "source", [chunk], [chunk], [[1.0]])

    assert connection.transactions == 1
    deletions = [query for query, _ in connection.queries if query.startswith("DELETE")]
    assert len(deletions) == 1
    assert "namespace = %s AND source_id = %s" in deletions[0]


def test_openai_embedder_restores_response_order(monkeypatch) -> None:
    response = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[2.0]),
            SimpleNamespace(index=0, embedding=[1.0]),
        ]
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **_: response))
    monkeypatch.setattr(openai, "OpenAI", lambda **_: client)

    assert OpenAIEmbedder("synthetic", "test-model").embed(["first", "second"]) == [
        [1.0], [2.0]
    ]


@pytest.mark.parametrize("indexes", [[0, 0], [0, 2]])
def test_openai_embedder_rejects_duplicate_or_missing_indexes(monkeypatch, indexes) -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=[1.0]) for index in indexes]
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **_: response))
    monkeypatch.setattr(openai, "OpenAI", lambda **_: client)

    with pytest.raises(ValueError, match="indexes"):
        OpenAIEmbedder("synthetic", "test-model").embed(["first", "second"])
