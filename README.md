# Vector Chunk Kit

Prepare human-reviewable text chunks, then embed and upsert them into a PostgreSQL database with pgvector. The input is your own JSONL file. The default command path is entirely local; external calls happen only when you explicitly run `ingest --apply`.

The project shows a small, inspectable pipeline: strict input validation, configurable chunk context, deterministic IDs, namespace checks, batched embedding requests, idempotent database writes, and safe failure behavior. The included data is fictional.

## How it works

```text
your JSONL documents
        │
        ▼
prepare ──> reviewable JSONL chunks ──> ingest preview (offline)
                                           │
                                           ▼ explicit --apply
                                  embeddings API + pgvector
```

`prepare` splits text on word boundaries, optionally overlaps adjacent chunks, and renders each part through a template with `{title}`, `{keywords}`, and `{body}`. A content hash tracks changes; a stable chunk ID identifies the namespace, source, and position. On repeat ingestion of the same input, unchanged hashes skip the embedding call. Each database batch commits atomically, so an interrupted run can be repeated.

## Try it locally

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --locked --all-extras
uv run vector-chunk-kit prepare \
  --input examples/synthetic-documents.jsonl \
  --output output/chunks.jsonl \
  --namespace demo \
  --template-file examples/template.txt
uv run vector-chunk-kit ingest --input output/chunks.jsonl --namespace demo
```

With the included fictional documents, `prepare` reports 2 chunks and the offline preview reports an estimate of 97 tokens. The estimate is based on character count, not model tokenization. Neither command needs an API key or database. Inspect `output/chunks.jsonl` before any write. `prepare` refuses to overwrite an existing output file; choose a new output path for another run. Output files, local environments, and credentials are ignored by Git.

### Input format

Each non-empty line is one JSON object:

```json
{"id":"guide-1","namespace":"demo","title":"Example guide","text":"Your own source text goes here.","keywords":["example"],"metadata":{"category":"guide"}}
```

`id`, `namespace`, `title`, and `text` are required. `keywords` and `metadata` are optional. The requested `--namespace` must match every input record; mixed input fails before an output file is written. IDs and namespaces use letters, digits, `.`, `_`, `:`, or `-`. The tool rejects duplicate source IDs, empty text, malformed metadata, and unsupported template fields.

### Customize chunks

Make a text template containing `{body}` and optionally `{title}` and `{keywords}`. Use `--template-file PATH`, `--max-chars N`, and `--overlap-chars N` to control the output. Template fields are data substitution only; they do not execute code. This first version uses deterministic templates. It does not call an LLM to invent keywords or rewrite source material.

## Write to your own pgvector database

Use a disposable PostgreSQL database with the pgvector extension. Review and apply [schema.sql](schema.sql) to a fresh database yourself; the CLI never creates or migrates tables. `CREATE TABLE IF NOT EXISTS` does not upgrade an older table. A namespace containing chunks without a model record is rejected; regenerate its embeddings in a new namespace or review a migration yourself. Stop older ingesters before migrating. Supply `OPENAI_API_KEY` and `DATABASE_URL` through your environment or a local secret manager. [.env.example](.env.example) contains placeholders only. The CLI does not load `.env` files automatically.

```bash
uv run vector-chunk-kit ingest \
  --input output/chunks.jsonl \
  --namespace demo \
  --batch-size 16 \
  --apply \
  --confirm-namespace demo
```

The selected namespace must be repeated exactly with `--confirm-namespace`. The optional `--model` defaults to `text-embedding-3-small`. The first applied run binds its namespace to one embedding model and vector dimension in the database. A later run with a different model fails before requesting embeddings; a different dimension fails before writing that batch. Use a new namespace and a separate retrieval configuration when changing models. The [OpenAI embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create) processes the chunk text, so send only data you are allowed to share with that provider. Costs depend on the model and input; the preview is **not** a price quote. The database adapter uses [pgvector's Psycopg integration](https://github.com/pgvector/pgvector-python#psycopg-3).

## Verify

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
uv build
```

The tests cover deterministic output, custom context, malformed input, cross-namespace rejection, model and dimension guards, idempotent skips, batch failure, invalid vectors, and a preview that works without credentials. CI runs these checks on Linux and Windows, plus a separate integration job against a disposable PostgreSQL 16 service with pgvector. That job applies `schema.sql` in an isolated schema, checks a real insert and update, and rejects model and dimension changes. The local suite skips this test unless `CHUNKKIT_TEST_DATABASE_URL` points to a PostgreSQL instance on localhost named `chunkkit_test`. Never set that variable to an operational database.

## Boundaries

- The CLI scopes records and queries by namespace to prevent accidental mixing. It is **not** a full multi-tenant authorization service. Shared deployments need database roles, row-level security, and their own identity checks.
- Replacing a source with fewer chunks updates the surviving positions but does not delete old positions. Review or remove stale positions in your own data lifecycle before using this for retrieval.
- Concurrent ingestions of different revisions of one source use last-writer-wins updates. Serialize writers for a source if update order matters. Direct database writes can bypass the CLI's model and dimension checks.
- One `--apply` run may make billable embedding requests. Partial database failure can require retrying a batch; already committed chunks are skipped on a later run.
- This repository contains only synthetic examples. Do not commit prepared output containing private text or credentials.

See [architecture and security notes](docs/ARCHITECTURE.md) for the trust boundaries and failure model.
