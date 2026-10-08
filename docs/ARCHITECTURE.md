# Architecture and security

## Components

`prepare.py` validates developer-owned documents and creates deterministic chunks. `records.py` handles the JSONL boundary and verifies chunk IDs and content hashes when reading them back. `ingest.py` coordinates batched embedding and upsert through small provider/store interfaces. `adapters.py` supplies the optional OpenAI and pgvector implementations. `cli.py` keeps preview offline and gates writes behind explicit flags.

## Trust boundaries

1. **Input file:** Untrusted JSONL is parsed and validated. Every record must match the requested namespace. The tool never reads from an existing operational database to construct chunks.
2. **Prepared file:** This file can contain the full source text. Keep it private and inspect it before applying. IDs and hashes are recomputed on read so edited content cannot silently retain a previous hash.
3. **Embedding provider:** Only `ingest --apply` sends text outside the machine. The provider key comes from the environment and is never written to the prepared file. Provider/database exception details are suppressed in the CLI to avoid leaking secrets or source text into logs.
4. **Database:** The fixed tables and parameterized SQL avoid developer-controlled table names. The first applied run registers one embedding model per namespace in a transaction only if that namespace has no legacy chunks. Each batch checks its vector dimension against the namespace record before writing. Reads include a namespace predicate. Writes contain the namespace in the unique key and run inside a transaction per batch. The CLI never executes schema changes.

## Failure behavior

- Invalid or mixed-namespace documents fail before the output is opened.
- Invalid prepared hashes or IDs fail before external connections are made.
- A failed embedding call produces no write for that batch. A failed database statement rolls back its batch.
- A failed first run can leave an empty namespace model registration. Reuse that model or choose a new namespace; do not silently change a namespace's vector space.
- An older database with chunks but no model record fails closed. The schema file does not migrate existing tables; migrations need separate review with old writers stopped.
- Earlier committed batches remain after a later batch fails. Re-running skips chunks whose stored content hash matches. API calls for an uncommitted failed batch may be billed again.

## What this does not claim

Namespace labels alone do not authenticate a caller or enforce tenant isolation in a shared service. Deployments with mutually untrusted users need identity, role checks, PostgreSQL row-level security, and per-tenant operational controls. This project does not supply those policies. Concurrent writers for the same source use last-writer-wins semantics. Direct database writes can bypass CLI checks. The project also does not delete stale chunks when a document becomes shorter, estimate exact token billing, or verify a remote production configuration.
