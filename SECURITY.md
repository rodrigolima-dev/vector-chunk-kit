# Security

Use only documents and a database you control. Review generated chunks before `--apply`; prepared JSONL contains source text. Keep credentials in a local secret manager or environment, and never add them to issues, logs, examples, or commits.

The included namespace checks protect against accidental mixing in this CLI. They do not replace authorization or row-level security in a shared application. See [architecture and security notes](docs/ARCHITECTURE.md) for exact boundaries.

If you discover a vulnerability, use GitHub private vulnerability reporting when available. Do not post secrets or private source data in a public issue.
