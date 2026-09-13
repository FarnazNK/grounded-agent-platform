# Observability

The API emits structured JSON logs, request correlation IDs (`X-Request-ID`), Prometheus metrics for HTTP, retrieval, LLM latency, DB latency, token usage, cache behavior, and ingestion failures. The Grafana dashboard JSON in `monitoring/grafana/dashboards/rag-agent-overview.json` provides panels for throughput, p95 latency, errors, retrieval latency, token usage, and ingestion failures.
