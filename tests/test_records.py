from pathlib import Path

import pytest

from vector_chunk_kit.records import read_documents


def test_jsonl_rejects_nonstandard_nan_value(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    source.write_text('{"id":"x","metadata":{"score":NaN}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        read_documents(source)
