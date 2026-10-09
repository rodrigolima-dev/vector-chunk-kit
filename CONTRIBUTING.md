# Contributing

Small, focused fixes are welcome. Open an issue describing the behavior and a synthetic reproduction before larger changes. For a pull request:

1. Keep examples fictional and do not include provider keys, customer documents, prepared output, database exports, or real endpoint URLs.
2. Add a test that exercises the behavior at the relevant boundary, especially namespace checks, failures, and database writes.
3. Run `uv sync --locked --all-extras`, `uv run pytest -q`, `uv run ruff check .`, `uv run mypy src`, and `uv build`.
4. Explain the user-visible change and any provider cost or database effect in the pull request.

CI also runs against a disposable PostgreSQL service. For a local integration run, follow [the setup guide](docs/LOCAL_POSTGRES.md). Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
