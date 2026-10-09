"""Embed and upsert prepared chunks."""

import math
from dataclasses import dataclass
from typing import Protocol

from .prepare import SAFE_ID, Chunk, make_chunk_id, make_content_hash


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Store(Protocol):
    def ensure_model(self, namespace: str, model: str) -> None: ...

    def ensure_dimensions(self, namespace: str, dimensions: int) -> None: ...

    def existing_hashes(self, namespace: str, ids: list[str]) -> dict[str, str]: ...

    def upsert(self, items: list[Chunk], vectors: list[list[float]]) -> None: ...

    def replace_source(
        self,
        namespace: str,
        source_id: str,
        current: list[Chunk],
        pending: list[Chunk],
        vectors: list[list[float]],
    ) -> None: ...


@dataclass(frozen=True)
class IngestResult:
    total: int
    skipped: int
    written: int


def ingest_chunks(
    chunks: list[Chunk],
    namespace: str,
    embedder: Embedder,
    store: Store,
    batch_size: int,
    model: str = "text-embedding-3-small",
    replace_source: str | None = None,
) -> IngestResult:
    if not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be between 1 and 128")
    if not model.strip():
        raise ValueError("embedding model is required")
    ids: set[str] = set()
    for chunk in chunks:
        if chunk.namespace != namespace:
            raise ValueError("chunk namespace does not match requested namespace")
        if not chunk.text or make_content_hash(chunk.text, chunk.metadata) != chunk.content_hash:
            raise ValueError("chunk content hash is invalid")
        if chunk.chunk_id != make_chunk_id(namespace, chunk.source_id, chunk.chunk_index):
            raise ValueError("chunk id does not match its source")
        if chunk.chunk_id in ids:
            raise ValueError("duplicate chunk id")
        ids.add(chunk.chunk_id)

    if replace_source is not None:
        validate_replacement(chunks, namespace, replace_source)

    store.ensure_model(namespace, model)

    skipped = 0
    written = 0
    staged: list[Chunk] = []
    staged_vectors: list[list[float]] = []
    staged_dimensions: int | None = None
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        existing = store.existing_hashes(namespace, [chunk.chunk_id for chunk in batch])
        pending = [chunk for chunk in batch if existing.get(chunk.chunk_id) != chunk.content_hash]
        skipped += len(batch) - len(pending)
        if not pending:
            continue
        vectors = embedder.embed([chunk.text for chunk in pending])
        if len(vectors) != len(pending) or not vectors:
            raise ValueError("embedding provider returned an invalid vector count")
        dimensions = _validate_vectors(vectors)
        if replace_source is not None:
            if staged_dimensions is not None and dimensions != staged_dimensions:
                raise ValueError("embedding provider changed vector dimension between batches")
            staged_dimensions = dimensions
            staged.extend(pending)
            staged_vectors.extend(vectors)
            continue
        store.ensure_dimensions(namespace, dimensions)
        store.upsert(pending, vectors)
        written += len(pending)
    if replace_source is not None:
        store.replace_source(namespace, replace_source, chunks, staged, staged_vectors)
        written = len(staged)
    return IngestResult(total=len(chunks), skipped=skipped, written=written)


def validate_replacement(chunks: list[Chunk], namespace: str, source_id: str) -> None:
    if not SAFE_ID.fullmatch(source_id):
        raise ValueError("replace-source must be a simple non-empty identifier")
    if not chunks or any(
        chunk.namespace != namespace or chunk.source_id != source_id for chunk in chunks
    ):
        raise ValueError("replacement requires chunks from exactly the requested source")
    if sorted(chunk.chunk_index for chunk in chunks) != list(range(len(chunks))):
        raise ValueError("replacement requires contiguous chunk indexes starting at zero")


def _validate_vectors(vectors: list[list[float]]) -> int:
    dimensions = len(vectors[0])
    if not 1 <= dimensions <= 4096 or any(
        len(vector) != dimensions
        or any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise ValueError("embedding provider returned an invalid vector")
    return dimensions
