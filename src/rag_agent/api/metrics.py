from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

REQUESTS = Counter('rag_agent_requests_total', 'Total HTTP requests.', labelnames=('endpoint', 'status'), registry=REGISTRY)
REQUEST_LATENCY = Histogram('rag_agent_request_latency_seconds', 'End-to-end request latency.', labelnames=('endpoint',), buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10), registry=REGISTRY)
RETRIEVAL_LATENCY = Histogram('rag_agent_retrieval_latency_seconds', 'Retrieval latency.', buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1), registry=REGISTRY)
LLM_LATENCY = Histogram('rag_agent_llm_latency_seconds', 'LLM latency.', buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10), registry=REGISTRY)
DB_LATENCY = Histogram('rag_agent_db_latency_seconds', 'Database latency for retrieval paths.', buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5), registry=REGISTRY)
ERRORS = Counter('rag_agent_errors_total', 'Errors by code.', labelnames=('code',), registry=REGISTRY)
TOKENS = Counter('rag_agent_tokens_total', 'Token usage.', labelnames=('type',), registry=REGISTRY)
CACHE_HITS = Counter('rag_agent_cache_hits_total', 'Cache hits by cache name.', labelnames=('cache',), registry=REGISTRY)
CACHE_MISSES = Counter('rag_agent_cache_misses_total', 'Cache misses by cache name.', labelnames=('cache',), registry=REGISTRY)
INGESTION_JOBS = Counter('rag_agent_ingestion_jobs_total', 'Ingestion jobs by operation and status.', labelnames=('operation', 'status'), registry=REGISTRY)
INGESTION_FAILURES = Counter('rag_agent_ingestion_failures_total', 'Ingestion failures by reason.', labelnames=('reason',), registry=REGISTRY)
RETRIEVAL_GROUNDEDNESS = Gauge('rag_agent_retrieval_groundedness_ratio', 'Latest groundedness ratio observed by the API.', registry=REGISTRY)
ACTIVE_REQUESTS = Gauge('rag_agent_active_requests', 'Concurrent in-flight requests.', registry=REGISTRY)


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)
