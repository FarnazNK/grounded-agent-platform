# Architecture

The service is a FastAPI API backed by PostgreSQL + pgvector. Every request carries a JWT identity, resolves organization/workspace membership, and scopes ingestion and retrieval to that workspace. Uploads are parsed, chunked, embedded, and persisted with ingestion-job state transitions. Retrieval combines pgvector dense search with PostgreSQL full-text ranking and fuses both lists before answer generation.
