# Architecture and security boundaries

```text
JSONL documents -> validate -> prepare -> reviewable JSONL -> offline preview
                                               |
                                               +-> explicit apply -> embeddings API -> PostgreSQL/pgvector
```

## Design decisions

| Decision | Reason | Tradeoff |
| --- | --- | --- |
| Keep `prepare` and preview offline | A developer can inspect every rendered chunk before a provider sees it | Source text exists in a local output file and must be protected |
| Build deterministic IDs from namespace, source, and position | Repeated imports target the same database row | A shorter source leaves surplus positions unless explicitly replaced |
| Hash rendered text and metadata | Detect changed records and skip unchanged embeddings | Metadata-only edits still request a new embedding |
| Bind one model and dimension to each namespace | Avoid mixing incompatible vector spaces through this CLI | Model changes need a new namespace and matching retrieval configuration |
| Offer incremental and explicit source replacement modes | Routine updates can be batched; a single source can be made complete | Replacement holds a transaction for one source and requires a one-source file |

`records.py` validates JSONL and recomputes IDs and hashes when prepared chunks are read. `prepare.py` splits text and applies a constrained template. `ingest.py` coordinates the provider and store through small interfaces; `adapters.py` implements OpenAI and PostgreSQL. `cli.py` requires `--apply` and exact namespace confirmation before writing.

## Trust boundaries

1. **Input:** Every record must match the requested namespace. Malformed input fails before an output file is opened. The tool never extracts text from an operational database.
2. **Prepared file:** It contains source text. Recomputed IDs and hashes catch accidental mismatch between a record and its generated fields. The hash is not a signature or tamper protection. Keep the file private and review it before apply.
3. **Provider:** Only `ingest --apply` sends rendered text to the embeddings API. The API key comes from the environment. Provider and database exception details are suppressed at the CLI boundary to avoid printing sensitive payloads or credentials.
4. **Database:** Queries use fixed tables, parameterized values, and namespace predicates. The first applied run registers a model for the namespace only when there are no legacy chunks lacking that record. Dimensions are checked before writes. The CLI does not apply `schema.sql` or migrate an existing database.

## Write and failure model

- Incremental mode embeds and commits in batches. A failed provider call writes nothing for that batch; a failed SQL statement rolls back that batch. Earlier batches remain and a retry skips rows with matching hashes.
- `--replace-source` requires a file for one source with positions `0..n-1`. It gathers pending embeddings before changing chunks, then upserts and deletes surplus positions in one transaction for that source. Provider failure leaves stored chunks unchanged; SQL failure rolls the source transaction back.
- A failed first apply may leave an empty namespace model registration. Reuse the same model or choose a new namespace. An older database containing chunks without a model record fails closed; applying `schema.sql` with `IF NOT EXISTS` does not upgrade it.
- Provider requests may be billed even if the following database transaction fails. Two concurrent writers of the same source can still overwrite each other; serialize writers when update order matters.

Namespace labels and CLI checks do **not** authenticate users or enforce tenant separation in a shared service. A multi-user deployment needs identity checks, PostgreSQL roles, row-level security, and a policy for readers, writers, and migrations. Direct SQL writes bypass this CLI's model and dimension checks.
