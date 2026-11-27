# ADR 0009: Shared rate limits and cache, and an audit event stream

## Context

Rate limits and the retrieval cache lived in each process's memory. With more than one API
replica, each replica enforced its own limits (so the real limit grew with the replica count)
and kept its own cache. The login limiter's per-IP table also never shrank, so a flood of
distinct addresses grew memory without bound. There was no record of who retrieved which
documents.

## Decision

- **Optional Redis/Valkey** (`APP_REDIS_URL`) backs rate limits and the retrieval cache. Without
  it, behaviour is unchanged, which keeps tests and single-replica deployments simple.
- **Fixed-window counters** in Redis: one `INCR` + `EXPIRE` per request. Simpler and cheaper than a
  sliding log; worst case allows up to 2x the limit across a window boundary, which is acceptable
  for abuse protection.
- **Degrade, don't fail.** If Redis is down, rate limiting falls back to per-replica memory and
  the cache becomes a miss. Failing open would remove brute-force protection; failing closed
  would turn a cache outage into an API outage.
- **No sensitive data in Redis key names.** Rate-limit keys (user IDs, IPs) and the query part of
  cache keys are SHA-256 hashed. Cache values still hold tenant answers, so production should use
  `rediss://` (TLS) and a password; the app logs a warning otherwise.
- **Cache safety does not depend on invalidation.** Cache keys include the workspace
  `index_version`, which is bumped on every ingest and delete, so a failed invalidation cannot
  serve content from a deleted document.
- **Audit events as structured logs** (`rag_agent.audit` logger, `"audit": true`). They record login
  success and failure, bootstrap, upload, reindex, delete, each query with the source documents
  disclosed, and every authentication, authorization or rate-limit denial. Query text and emails
  are logged only as fingerprints. Shipping logs to a SIEM or WORM bucket is an operator choice
  and needs no new service in the app.
- **X-Forwarded-For is ignored by default** (`APP_TRUSTED_PROXY_COUNT=0`), because a client can
  set it to any value and dodge per-IP limits.

## Consequences

- Horizontal scaling no longer weakens rate limits.
- A database-backed, hash-chained audit table (tamper evidence, in-app audit viewer) is the next
  step if logs alone are not enough for a compliance requirement.
