# ADR 0002: pgvector as primary vector backend
pgvector keeps embeddings, metadata, and tenant filters in one transactional datastore. This simplifies multi-tenant authorization, migrations, operational backups, and integration testing.
