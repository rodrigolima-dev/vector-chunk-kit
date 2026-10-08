import pytest

from vector_chunk_kit.ingest import ingest_chunks
from vector_chunk_kit.prepare import Chunk, prepare_documents


def chunks() -> list[Chunk]:
    return prepare_documents(
        [{"id": "doc", "namespace": "demo", "title": "Demo", "text": "one two three four"}],
        "demo",
        7,
        0,
        "{body}",
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, unchanged: dict[str, str] | None = None) -> None:
        self.unchanged = unchanged or {}
        self.reads: list[tuple[str, list[str]]] = []
        self.writes: list[list[Chunk]] = []

    def existing_hashes(self, namespace: str, ids: list[str]) -> dict[str, str]:
        self.reads.append((namespace, ids))
        return self.unchanged

    def upsert(self, items: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(items) == len(vectors)
        self.writes.append(items)


def test_ingest_is_scoped_batched_and_skips_unchanged_content() -> None:
    items = chunks()
    store = FakeStore({items[0].chunk_id: items[0].content_hash})
    embedder = FakeEmbedder()

    result = ingest_chunks(items, "demo", embedder, store, batch_size=2)

    assert result.total == len(items)
    assert result.skipped == 1
    assert result.written == len(items) - 1
    assert all(namespace == "demo" for namespace, _ in store.reads)
    assert all(len(batch) <= 2 for batch in embedder.calls)
    assert sum(map(len, store.writes)) == len(items) - 1


def test_ingest_rejects_cross_namespace_before_external_calls() -> None:
    item = chunks()[0]
    embedder = FakeEmbedder()
    store = FakeStore()

    with pytest.raises(ValueError, match="namespace"):
        ingest_chunks([item], "another", embedder, store, batch_size=2)

    assert store.reads == []
    assert embedder.calls == []


def test_ingest_does_not_write_a_failed_embedding_batch() -> None:
    class FailingEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("provider failed")

    store = FakeStore()
    with pytest.raises(RuntimeError, match="provider failed"):
        ingest_chunks(chunks(), "demo", FailingEmbedder(), store, batch_size=2)
    assert store.writes == []


def test_ingest_rejects_invalid_vector_shape() -> None:
    class BadEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[float("nan")]] * len(texts)

    store = FakeStore()
    with pytest.raises(ValueError, match="vector"):
        ingest_chunks(chunks(), "demo", BadEmbedder(), store, batch_size=2)
    assert store.writes == []
