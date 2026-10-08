from contextlib import nullcontext

import pgvector.psycopg
import psycopg
import pytest

from vector_chunk_kit.adapters import PgvectorStore
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
