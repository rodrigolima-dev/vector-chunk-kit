import pytest

from vector_chunk_kit.ingest import ingest_chunks
from vector_chunk_kit.prepare import Chunk, make_chunk_id, prepare_documents


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
        self.models: dict[str, str] = {}
        self.dimensions: dict[str, int] = {}

    def ensure_model(self, namespace: str, model: str) -> None:
        existing = self.models.setdefault(namespace, model)
        if existing != model:
            raise ValueError("namespace uses a different embedding model")

    def ensure_dimensions(self, namespace: str, dimensions: int) -> None:
        existing = self.dimensions.setdefault(namespace, dimensions)
        if existing != dimensions:
            raise ValueError("namespace uses a different vector dimension")

    def existing_hashes(self, namespace: str, ids: list[str]) -> dict[str, str]:
        self.reads.append((namespace, ids))
        return self.unchanged

    def upsert(self, items: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(items) == len(vectors)
        self.writes.append(items)

    def replace_source(
        self,
        namespace: str,
        source_id: str,
        current: list[Chunk],
        pending: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        assert namespace == "demo"
        assert source_id == "doc"
        assert len(pending) == len(vectors)
        if vectors:
            self.ensure_dimensions(namespace, len(vectors[0]))
        self.writes.append(pending)


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


def test_model_change_in_existing_namespace_fails_before_embedding() -> None:
    items = chunks()
    store = FakeStore({items[0].chunk_id: items[0].content_hash})
    store.models["demo"] = "first-model"
    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="different embedding model"):
        ingest_chunks(items, "demo", embedder, store, batch_size=2, model="second-model")

    assert store.reads == []
    assert store.writes == []
    assert embedder.calls == []


def test_dimension_change_between_batches_fails_before_second_write() -> None:
    class ChangingEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            dimensions = 2 if len(self.calls) == 1 else 3
            return [[1.0] * dimensions for _ in texts]

    store = FakeStore()
    embedder = ChangingEmbedder()

    with pytest.raises(ValueError, match="different vector dimension"):
        ingest_chunks(chunks(), "demo", embedder, store, batch_size=1)

    assert len(store.writes) == 1
    assert store.dimensions["demo"] == 2


def test_replace_source_embeds_all_batches_before_one_store_write() -> None:
    items = chunks()
    store = FakeStore()
    embedder = FakeEmbedder()

    result = ingest_chunks(
        items, "demo", embedder, store, batch_size=1, replace_source="doc"
    )

    assert result.written == len(items)
    assert len(embedder.calls) == len(items)
    assert store.writes == [items]


def test_replace_source_rejects_other_sources_and_missing_indexes_before_store_calls() -> None:
    items = chunks()
    other_source = Chunk(
        **{
            **vars(items[0]),
            "source_id": "other",
            "chunk_id": make_chunk_id("demo", "other", items[0].chunk_index),
        }
    )
    store = FakeStore()
    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="source"):
        ingest_chunks(items + [other_source], "demo", embedder, store, 1, replace_source="doc")
    with pytest.raises(ValueError, match="contiguous"):
        ingest_chunks(items[1:], "demo", embedder, store, 1, replace_source="doc")

    assert store.models == {}
    assert store.reads == []
    assert store.writes == []
    assert embedder.calls == []


def test_replace_source_embedding_failure_does_not_write() -> None:
    class FailingSecondBatch(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            if len(self.calls) == 2:
                raise RuntimeError("provider failed")
            return [[1.0, 0.0, 0.0] for _ in texts]

    store = FakeStore()
    with pytest.raises(RuntimeError, match="provider failed"):
        ingest_chunks(chunks(), "demo", FailingSecondBatch(), store, 1, replace_source="doc")
    assert store.writes == []


def test_replace_source_skips_unchanged_vectors_but_still_finalizes_source() -> None:
    items = chunks()
    store = FakeStore({item.chunk_id: item.content_hash for item in items})
    embedder = FakeEmbedder()

    result = ingest_chunks(items, "demo", embedder, store, 1, replace_source="doc")

    assert result.skipped == len(items)
    assert result.written == 0
    assert embedder.calls == []
    assert store.writes == [[]]


def test_replace_source_keeps_model_and_dimension_guards() -> None:
    items = chunks()
    store = FakeStore()
    store.models["demo"] = "first-model"
    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="different embedding model"):
        ingest_chunks(items, "demo", embedder, store, 1, "second-model", "doc")
    assert embedder.calls == []
    assert store.writes == []

    store.dimensions["demo"] = 2
    with pytest.raises(ValueError, match="different vector dimension"):
        ingest_chunks(items, "demo", embedder, store, 1, "first-model", "doc")
    assert store.writes == []


def test_replace_source_rejects_dimension_change_between_batches_before_write() -> None:
    class ChangingEmbedder(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return [[1.0] * (2 if len(self.calls) == 1 else 3) for _ in texts]

    store = FakeStore()
    with pytest.raises(ValueError, match="dimension between batches"):
        ingest_chunks(chunks(), "demo", ChangingEmbedder(), store, 1, replace_source="doc")
    assert store.writes == []
