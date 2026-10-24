# Trade-offs and limitations

## Current limits
- The default deployment uses synchronous provider calls; high-throughput production should move LLM and embedding calls to async-native SDKs or worker pools.
- The built-in rate limiter is process-local; multi-instance production should externalize it.
- Large binary formats beyond PDF are deliberately out of scope.
- The pgvector schema currently uses 256-dimensional embeddings; changing dimensions requires a migration and full re-index.
- Offline evals run deterministic answers and embeddings, so they measure retrieval, ranking and citation plumbing, not answer quality. Answer quality comes from `scripts/run_model_eval.py` (real model + LLM judge), which isn't run in CI because it costs API credits.
- The offline eval store uses BM25 without stemming; production PostgreSQL full-text search stems, so offline keyword ranking is somewhat weaker than production's.
- The LLM judge is a model grading a model. It is checked against written criteria per case and unparseable verdicts count as failures, but its verdicts should be spot-checked.

## 10x plan
- Move rate limiting and request dedupe to Redis.
- Add async ingestion workers and object storage for originals.
- Partition large tenants by workspace-level routing keys.

## 100x plan
- Separate retrieval and generation services.
- Introduce queue-backed ingestion, background reindexing, and sharded pgvector tables.
- Add managed search for richer lexical ranking.

## Non-goals
- Multi-modal generation.
- Multi-modal or autonomous-agent features.
- Unsupported security claims without measured controls.
