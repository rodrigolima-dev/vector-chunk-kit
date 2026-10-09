import json
from pathlib import Path

from vector_chunk_kit.cli import main


def write_documents(path: Path, namespace: str = "demo") -> None:
    path.write_text(
        json.dumps(
            {
                "id": "library-guide",
                "namespace": namespace,
                "title": "Library guide",
                "text": "Visitors may borrow books for two weeks.",
                "keywords": ["books", "borrowing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepare_then_preview_needs_no_credentials_or_database(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "source.jsonl"
    prepared = tmp_path / "chunks.jsonl"
    write_documents(source)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert (
        main(["prepare", "--input", str(source), "--output", str(prepared), "--namespace", "demo"])
        == 0
    )
    assert "Prepared 1 chunk in" in capsys.readouterr().out
    chunks = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines()]
    assert len(chunks) == 1
    assert chunks[0]["namespace"] == "demo"
    assert main(["ingest", "--input", str(prepared), "--namespace", "demo"]) == 0
    assert "Preview: 1 chunk in" in capsys.readouterr().out


def test_apply_requires_exact_namespace_confirmation_before_credentials(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "source.jsonl"
    prepared = tmp_path / "chunks.jsonl"
    write_documents(source)
    assert (
        main(["prepare", "--input", str(source), "--output", str(prepared), "--namespace", "demo"])
        == 0
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert main(["ingest", "--input", str(prepared), "--namespace", "demo", "--apply"]) == 2
    assert "confirmation" in capsys.readouterr().err


def test_prepare_rejects_mixed_namespace_without_output(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.jsonl"
    prepared = tmp_path / "chunks.jsonl"
    write_documents(source, namespace="other")

    assert (
        main(["prepare", "--input", str(source), "--output", str(prepared), "--namespace", "demo"])
        == 2
    )
    assert not prepared.exists()
    assert "namespace" in capsys.readouterr().err


def test_invalid_chunk_content_hash_cannot_reach_ingestion(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.jsonl"
    prepared = tmp_path / "chunks.jsonl"
    write_documents(source)
    assert (
        main(["prepare", "--input", str(source), "--output", str(prepared), "--namespace", "demo"])
        == 0
    )
    chunk = json.loads(prepared.read_text(encoding="utf-8"))
    chunk["text"] = "Changed after preparation"
    prepared.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    assert main(["ingest", "--input", str(prepared), "--namespace", "demo"]) == 2
    assert "hash" in capsys.readouterr().err


def test_duplicate_prepared_chunks_fail_in_offline_preview(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.jsonl"
    prepared = tmp_path / "chunks.jsonl"
    write_documents(source)
    assert (
        main(["prepare", "--input", str(source), "--output", str(prepared), "--namespace", "demo"])
        == 0
    )
    one_line = prepared.read_text(encoding="utf-8")
    prepared.write_text(one_line + one_line, encoding="utf-8")

    assert main(["ingest", "--input", str(prepared), "--namespace", "demo"]) == 2
    assert "duplicate" in capsys.readouterr().err


def test_replace_source_requires_apply(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.jsonl"
    prepared = tmp_path / "chunks.jsonl"
    write_documents(source)
    assert (
        main(["prepare", "--input", str(source), "--output", str(prepared), "--namespace", "demo"])
        == 0
    )

    assert (
        main([
            "ingest", "--input", str(prepared), "--namespace", "demo",
            "--replace-source", "library-guide",
        ])
        == 2
    )
    assert "--apply" in capsys.readouterr().err


def test_replace_source_rejects_mixed_and_sparse_files_before_credentials(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "source.jsonl"
    prepared = tmp_path / "chunks.jsonl"
    write_documents(source)
    assert (
        main(["prepare", "--input", str(source), "--output", str(prepared), "--namespace", "demo"])
        == 0
    )
    original = json.loads(prepared.read_text(encoding="utf-8"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    command = [
        "ingest", "--input", str(prepared), "--namespace", "demo",
        "--replace-source", "library-guide", "--apply", "--confirm-namespace", "demo",
    ]

    from vector_chunk_kit.prepare import make_chunk_id

    other = {
        **original,
        "source_id": "another",
        "chunk_id": make_chunk_id("demo", "another", 0),
    }
    prepared.write_text(json.dumps(original) + "\n" + json.dumps(other) + "\n", encoding="utf-8")
    assert main(command) == 2
    assert "source" in capsys.readouterr().err

    sparse = {**original, "chunk_index": 1}
    sparse["chunk_id"] = make_chunk_id("demo", "library-guide", 1)
    prepared.write_text(json.dumps(sparse) + "\n", encoding="utf-8")
    assert main(command) == 2
    assert "contiguous" in capsys.readouterr().err
