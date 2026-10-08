"""Command-line interface with preview as the default ingestion mode."""

import argparse
import os
import sys
from pathlib import Path

from .ingest import ingest_chunks
from .prepare import DEFAULT_TEMPLATE, prepare_documents
from .records import read_chunks, read_documents, write_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and ingest reviewable text chunks")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Generate local chunks from JSONL documents")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--namespace", required=True)
    prepare.add_argument("--template-file", type=Path)
    prepare.add_argument("--max-chars", type=int, default=800)
    prepare.add_argument("--overlap-chars", type=int, default=80)
    ingest = commands.add_parser("ingest", help="Preview or explicitly write prepared chunks")
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument("--namespace", required=True)
    ingest.add_argument("--batch-size", type=int, default=16)
    ingest.add_argument("--model", default="text-embedding-3-small")
    ingest.add_argument("--apply", action="store_true")
    ingest.add_argument("--confirm-namespace")
    args = parser.parse_args(argv)

    try:
        if args.command == "prepare":
            if args.input.resolve() == args.output.resolve():
                raise ValueError("input and output paths must differ")
            template = (
                args.template_file.read_text(encoding="utf-8")
                if args.template_file
                else DEFAULT_TEMPLATE
            )
            chunks = prepare_documents(
                read_documents(args.input),
                args.namespace,
                args.max_chars,
                args.overlap_chars,
                template,
            )
            write_chunks(args.output, chunks)
            print(f"Prepared {len(chunks)} chunks in {args.output}")
            return 0

        chunks = read_chunks(args.input)
        if any(chunk.namespace != args.namespace for chunk in chunks):
            raise ValueError("prepared chunk namespace does not match requested namespace")
        if not args.apply:
            rough_tokens = sum((len(chunk.text) + 3) // 4 for chunk in chunks)
            print(
                f"Preview: {len(chunks)} chunks in namespace {args.namespace}; "
                f"rough character-based estimate: {rough_tokens} tokens. No external calls made."
            )
            return 0
        if args.confirm_namespace != args.namespace:
            raise ValueError("exact namespace confirmation is required with --apply")
        if not 1 <= args.batch_size <= 128:
            raise ValueError("batch-size must be between 1 and 128")
        api_key = os.environ.get("OPENAI_API_KEY")
        database_url = os.environ.get("DATABASE_URL")
        if not api_key or not database_url:
            raise ValueError("OPENAI_API_KEY and DATABASE_URL must be set for --apply")
        try:
            from .adapters import OpenAIEmbedder, PgvectorStore

            embedder = OpenAIEmbedder(api_key=api_key, model=args.model)
            with PgvectorStore(database_url) as store:
                result = ingest_chunks(chunks, args.namespace, embedder, store, args.batch_size)
        except Exception:
            print(
                "Ingestion failed. Provider and database details are suppressed; "
                "check local credentials and connectivity.",
                file=sys.stderr,
            )
            return 1
        print(f"Ingested {result.written}; skipped {result.skipped} unchanged chunks.")
        return 0
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
