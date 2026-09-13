# Trade-offs and limitations

## Current limits
- The default deployment uses synchronous provider calls; high-throughput production should move LLM and embedding calls to async-native SDKs or worker pools.
- The built-in rate limiter is process-local; multi-instance production should externalize it.
- Large binary formats beyond PDF are deliberately out of scope.

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
- Voice, image, or autonomous-agent features.
- Unsupported security claims without measured controls.
