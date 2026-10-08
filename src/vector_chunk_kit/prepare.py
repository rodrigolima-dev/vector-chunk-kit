"""Prepare deterministic text chunks."""

import hashlib
import json
import re
import string
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_TEMPLATE = "{title}\nKeywords: {keywords}\n\n{body}"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
ALLOWED_FIELDS = {"title", "keywords", "body"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content_hash: str
    namespace: str
    source_id: str
    chunk_index: int
    text: str
    metadata: dict[str, Any]


def prepare_documents(
    documents: Iterable[dict[str, object]],
    namespace: str,
    max_chars: int,
    overlap_chars: int,
    template: str = DEFAULT_TEMPLATE,
) -> list[Chunk]:
    if not SAFE_ID.fullmatch(namespace):
        raise ValueError("namespace must be a simple non-empty identifier")
    if not 1 <= max_chars <= 12000:
        raise ValueError("max_chars must be between 1 and 12000")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
    _validate_template(template)

    result: list[Chunk] = []
    source_ids: set[str] = set()
    for document in documents:
        source_id = document.get("id")
        if not isinstance(source_id, str) or not SAFE_ID.fullmatch(source_id):
            raise ValueError("id must be a simple non-empty identifier")
        if source_id in source_ids:
            raise ValueError("duplicate source id")
        source_ids.add(source_id)
        if document.get("namespace") != namespace:
            raise ValueError("document namespace does not match requested namespace")
        title = document.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be non-empty text")
        body = document.get("text")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("text must be non-empty")
        keywords = document.get("keywords", [])
        if not isinstance(keywords, list) or any(
            not isinstance(keyword, str) or not keyword.strip() for keyword in keywords
        ):
            raise ValueError("keywords must be a list of non-empty strings")
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
            raise ValueError("metadata must be a JSON object")
        try:
            clean_metadata: dict[str, Any] = json.loads(json.dumps(metadata, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must contain JSON values") from error

        for index, part in enumerate(split_text(body, max_chars, overlap_chars)):
            rendered = template.format(
                title=title.strip(), keywords=", ".join(keywords), body=part
            ).strip()
            if not rendered or len(rendered) > 16000:
                raise ValueError("rendered chunk must contain at most 16000 characters")
            result.append(
                Chunk(
                    chunk_id=make_chunk_id(namespace, source_id, index),
                    content_hash=make_content_hash(rendered, clean_metadata),
                    namespace=namespace,
                    source_id=source_id,
                    chunk_index=index,
                    text=rendered,
                    metadata=clean_metadata.copy(),
                )
            )
    return result


def make_chunk_id(namespace: str, source_id: str, index: int) -> str:
    key = f"{namespace}\0{source_id}\0{index}".encode()
    return hashlib.sha256(key).hexdigest()


def make_content_hash(text: str, metadata: dict[str, Any]) -> str:
    canonical_metadata = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256((text + "\0" + canonical_metadata).encode()).hexdigest()


def _validate_template(template: str) -> None:
    if not isinstance(template, str) or not template.strip():
        raise ValueError("template must be non-empty")
    fields: set[str] = set()
    try:
        for _, field, format_spec, conversion in string.Formatter().parse(template):
            if field is not None:
                if field not in ALLOWED_FIELDS or format_spec or conversion:
                    raise ValueError("template has an unsupported field or modifier")
                fields.add(field)
    except ValueError as error:
        raise ValueError("template is invalid") from error
    if "body" not in fields:
        raise ValueError("template must include {body}")


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    words = text.split()
    if any(len(word) > max_chars for word in words):
        raise ValueError("text contains a word longer than max_chars")
    parts: list[str] = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while (
            end < len(words) and length + len(words[end]) + (1 if end > start else 0) <= max_chars
        ):
            length += len(words[end]) + (1 if end > start else 0)
            end += 1
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        next_start = end
        if overlap_chars:
            tail_length = 0
            while next_start > start + 1:
                candidate = len(words[next_start - 1]) + (1 if tail_length else 0)
                if tail_length + candidate > overlap_chars:
                    break
                tail_length += candidate
                next_start -= 1
        start = next_start
    return parts
