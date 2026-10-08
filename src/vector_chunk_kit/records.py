"""Read and write the tool's JSONL interchange format."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

from .prepare import SAFE_ID, Chunk, make_chunk_id, make_content_hash


def read_documents(path: Path) -> list[dict[str, object]]:
    return [_object(record) for record in _read_jsonl(path)]


def read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    expected = set(Chunk.__dataclass_fields__)
    for record in _read_jsonl(path):
        item = _object(record)
        if set(item) != expected:
            raise ValueError("prepared chunk has an invalid structure")
        namespace, source_id, index = item["namespace"], item["source_id"], item["chunk_index"]
        text, content_hash, chunk_id = item["text"], item["content_hash"], item["chunk_id"]
        metadata = item["metadata"]
        if (
            not isinstance(namespace, str)
            or not SAFE_ID.fullmatch(namespace)
            or not isinstance(source_id, str)
            or not SAFE_ID.fullmatch(source_id)
            or type(index) is not int
            or index < 0
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(metadata, dict)
            or any(not isinstance(key, str) for key in metadata)
        ):
            raise ValueError("prepared chunk contains invalid fields")
        if chunk_id != make_chunk_id(namespace, source_id, index):
            raise ValueError("prepared chunk id does not match its source")
        if content_hash != make_content_hash(text, metadata):
            raise ValueError("prepared chunk content hash does not match text")
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                content_hash=content_hash,
                namespace=namespace,
                source_id=source_id,
                chunk_index=index,
                text=text,
                metadata=metadata,
            )
        )
    return chunks


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if len(line) > 2_000_000:
                raise ValueError(f"JSONL line {line_number} is too large")
            try:
                records.append(json.loads(line, parse_constant=_reject_constant))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"JSONL line {line_number} is invalid") from error
    if not records:
        raise ValueError("JSONL input has no records")
    return records


def _object(value: Any) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("JSONL records must be objects")
    return value


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"nonstandard JSON constant: {value}")
