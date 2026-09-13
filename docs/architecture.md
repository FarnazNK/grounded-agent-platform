# Architecture

The service is a FastAPI API backed by PostgreSQL + pgvector. Every request carries a JWT identity, resolves workspace membership, and scopes ingestion and retrieval to that workspace.

Uploads are size/type checked, parsed, chunked, embedded, and persisted with ingestion-job state transitions. Query retrieval combines pgvector dense search with PostgreSQL full-text search. Weighted reciprocal-rank fusion gives lexical matches slightly more influence so exact policy names, acronyms, and identifiers are not drowned out by noisy dense matches.

Retrieved text is treated as untrusted data. Input guardrails run on user queries, retrieved chunks containing obvious prompt-injection patterns are excluded before generation, and output checks redact PII and validate citations against the retrieved source set.
