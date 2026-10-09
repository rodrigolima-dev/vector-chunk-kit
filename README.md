# Vector Chunk Kit

Turn your own JSONL documents into reviewable, deterministic text chunks, then load them into PostgreSQL with pgvector. Preparation and preview run offline. An explicit `--apply` is required to call the embeddings API or write to a database.

For example, the included fictional garden guide starts as one document with a title, keywords, body, and metadata. `prepare` splits its body when needed and renders a chunk like this:

```text
Topic: Community garden guide
Keywords: seedlings, watering

Plant seedlings in loose soil. Water gently in the morning. Check the soil before watering again. Keep paths clear for visitors.
```

The resulting JSONL also carries a stable ID, source ID, position, content hash, and metadata. You can inspect the result before sending any text to a provider.

## Run the offline path

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked --all-extras
uv run vector-chunk-kit prepare --input examples/synthetic-documents.jsonl --output output/chunks.jsonl --namespace demo --template-file examples/template.txt
uv run vector-chunk-kit ingest --input output/chunks.jsonl --namespace demo
```

The example prepares two chunks and previews a rough estimate of 97 tokens. The estimate uses character count, not the embedding model's tokenizer or price. These commands need no API key or database. `prepare` refuses to overwrite an existing output file; remove that local file or select a different output path before repeating the example.

Each input line is one JSON object:

```json
{"id":"guide-1","namespace":"demo","title":"Example guide","text":"Your own source text goes here.","keywords":["example"],"metadata":{"category":"guide"}}
```

`id`, `namespace`, `title`, and `text` are required. `keywords` and `metadata` are optional. Records must all match `--namespace`; duplicate source IDs, empty text, invalid metadata, and malformed JSON fail before output is written. IDs and namespaces accept letters, digits, `.`, `_`, `:`, and `-`.

### Control chunk content

`--template-file` accepts a text template with `{body}` and optional `{title}` and `{keywords}` fields. `--max-chars` limits each **body part** before the template is rendered, while `--overlap-chars` repeats context between adjacent parts. The final rendered chunk can be longer than `--max-chars` because it also contains the title and keywords. Templates substitute data; they do not execute code. This project does not generate text or keywords with an LLM.

## Load into a disposable pgvector database

Follow [the local PostgreSQL walkthrough](docs/LOCAL_POSTGRES.md) to start a disposable database and apply [schema.sql](schema.sql). The CLI never creates tables or runs migrations. Set `OPENAI_API_KEY` and `DATABASE_URL` in your environment or local secret manager; [.env.example](.env.example) is placeholders only, and the CLI does not load `.env` automatically. Review the prepared file before applying: it contains the source text, and `--apply` sends chunk text to the [OpenAI embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create).

```bash
uv run vector-chunk-kit ingest --input output/chunks.jsonl --namespace demo --batch-size 16 --apply --confirm-namespace demo
```

The default is a read-only, offline preview. `--apply` requires the namespace to be repeated exactly in `--confirm-namespace`. The optional model defaults to `text-embedding-3-small`; a namespace is bound to one model and vector dimension on its first applied run. A model change fails before an embedding request, and a dimension mismatch fails before the affected write. Use a new namespace and matching retrieval configuration when changing models.

Incremental ingestion updates chunks at matching source positions and skips stored content hashes that are unchanged. It **does not remove** old positions when a source becomes shorter. The included updated garden guide shows how to replace one source completely:

```bash
uv run vector-chunk-kit prepare --input examples/synthetic-updated-garden.jsonl --output output/one-source.jsonl --namespace demo --template-file examples/template.txt
uv run vector-chunk-kit ingest --input output/one-source.jsonl --namespace demo
uv run vector-chunk-kit ingest --input output/one-source.jsonl --namespace demo --replace-source garden-guide --apply --confirm-namespace demo
```

Replacement requires exactly one source and consecutive positions starting at zero. It computes pending embeddings first, then writes updated positions and removes surplus positions in one database transaction for that source. A provider failure therefore leaves its stored chunks unchanged; a database failure rolls the source transaction back. API requests may still be billed even when a later database step fails. Repeating the same prepared input skips unchanged hashes. Different concurrent revisions of a source still require the caller to serialize writers.

The content hash includes rendered text and metadata. Changing metadata alone currently requests a new embedding even if the rendered text is identical. This is deliberate for simple change tracking, but it can add provider cost.

## Verify

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
uv build
```

CI runs unit checks on Linux and Windows with Python 3.11 and 3.14, plus PostgreSQL integration checks against a disposable pgvector service. The integration test runs locally only when `CHUNKKIT_TEST_DATABASE_URL` points to the expected loopback `chunkkit_test` database for user `chunkkit`; it rejects connection-routing overrides. See [the local PostgreSQL walkthrough](docs/LOCAL_POSTGRES.md) for the setup. The tests cover input validation, namespace and model guards, deterministic output, idempotence, failures, and source replacement.

## Design and limits

- The namespace is a data boundary in this CLI, **not** authentication or full tenant isolation. A shared service needs its own identity checks, database roles, row-level security, and operational controls.
- The prepared output may contain private text. It is ignored by Git under `output/`; keep other output paths private too. Never commit credentials, customer documents, or real database exports.
- An applied run can incur embeddings charges. The preview is an estimate, not a quote. Direct database writes bypass this CLI's model and dimension checks.
- Concurrent writers of one source are last-writer-wins. The tool does not coordinate readers or provide a production migration strategy.

Read [architecture and failure behavior](docs/ARCHITECTURE.md), [security reporting](SECURITY.md), and [contribution guidance](CONTRIBUTING.md) for the precise boundaries. All examples in this repository are fictional.
