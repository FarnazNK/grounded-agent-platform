# ADR 0001: Hybrid retrieval vs dense-only
Dense retrieval alone underperforms on exact-policy and acronym queries. The service keeps dense pgvector search and PostgreSQL lexical ranking, then fuses both so measurable recall does not depend on a single retrieval signal.
