# Migration notes

This release removes Chroma and all voice-serving code. Pull the latest `main`, recreate your environment, start PostgreSQL/pgvector, copy `.env.example` to `.env`, run `alembic upgrade head`, then bootstrap a workspace with `rag-agent bootstrap-demo` or `POST /v1/auth/bootstrap`. Existing local Chroma state is not migrated.
