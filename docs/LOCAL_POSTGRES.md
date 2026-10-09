# Disposable PostgreSQL setup

Use this only with a new local database. These commands match the PostgreSQL and pgvector image used by CI. The example password is for a container bound to your computer's loopback interface; choose another password if other local users can access Docker.

From the repository root, with Docker running:

```text
docker run --name chunkkit-pg --rm -d -e POSTGRES_USER=chunkkit -e POSTGRES_PASSWORD=test-only-password -e POSTGRES_DB=chunkkit_test -p 127.0.0.1:5432:5432 pgvector/pgvector:0.8.6-pg16
docker exec chunkkit-pg pg_isready -U chunkkit -d chunkkit_test
docker cp schema.sql chunkkit-pg:/tmp/schema.sql
docker exec chunkkit-pg psql -v ON_ERROR_STOP=1 -U chunkkit -d chunkkit_test -f /tmp/schema.sql
```

If `pg_isready` reports that the server is starting, repeat it before applying the schema. The container has no mounted data volume; stopping it deletes the database. Do not use a remote Docker context or publish the port on a public interface.

Set the local database URL in your shell. The separate test variable enables the PostgreSQL integration test:

```bash
export DATABASE_URL='postgresql://chunkkit:test-only-password@127.0.0.1:5432/chunkkit_test'
export CHUNKKIT_TEST_DATABASE_URL="$DATABASE_URL"
uv run pytest -q tests/test_postgres_integration.py
```

In PowerShell, use:

```powershell
$env:DATABASE_URL = 'postgresql://chunkkit:test-only-password@127.0.0.1:5432/chunkkit_test'
$env:CHUNKKIT_TEST_DATABASE_URL = $env:DATABASE_URL
uv run pytest -q tests/test_postgres_integration.py
```

The integration test creates temporary database objects and requires that exact local database name, user, and loopback address. To run `ingest --apply`, set `OPENAI_API_KEY` separately in the same shell and use only synthetic or otherwise authorized text. API calls may incur a charge. The integration test itself uses a fake embedder and does not call the API.

When finished:

```bash
docker stop chunkkit-pg
```

Unset the two database environment variables and `OPENAI_API_KEY` in the shell after testing. Never point this setup or its integration tests at an operational database.
