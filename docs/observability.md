# Observability

The API emits structured JSON logs, request correlation IDs (`X-Request-ID`), Prometheus metrics for HTTP, retrieval, LLM latency, DB latency, token usage, cache behavior, and ingestion failures. The Grafana dashboard JSON in `monitoring/grafana/dashboards/rag-agent-overview.json` provides panels for throughput, p95 latency, errors, retrieval latency, token usage, and ingestion failures.

## LLM tracing with Langfuse

Prometheus answers "is the service healthy?"; Langfuse answers "what did the model and the
agents actually do, and what did it cost?". Tracing is implemented in
`src/rag_agent/tracing.py` behind a small `Tracer` interface, so application code never
imports the Langfuse SDK directly.

### Run Langfuse locally (free)

```bash
docker compose -f monitoring/langfuse/docker-compose.yml up -d   # or: make langfuse-up
```

Open http://localhost:3001 and sign in as `admin@example.com` / `local-dev-password`. The
compose file bootstraps a project with the API keys `pk-lf-local-dev` / `sk-lf-local-dev`.
All default secrets in that file are local-only placeholders marked `CHANGEME`; override
them in `monitoring/langfuse/.env` before running it anywhere shared. Ports are remapped
(web 3001, MinIO 9092) so it runs alongside this repo's Grafana and Prometheus.

Langfuse Cloud's free Hobby tier works the same way: set `LANGFUSE_BASE_URL` to your cloud
region and use that project's keys.

### Enable it in the app

| Variable | Value |
|---|---|
| `APP_TRACING_BACKEND` | `langfuse` (default `none`) |
| `LANGFUSE_BASE_URL` | `http://localhost:3001`, or `http://host.docker.internal:3001` from the API container |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | project API keys |
| `APP_TRACING_RELEASE` | optional release/version label on every trace |

Install the SDK with `pip install -e '.[tracing]'`.

### What is traced

| Trace | Observations | Score |
|---|---|---|
| `rag.query` (user id, request id as session, workspace id) | `retrieve` with sources and dense/sparse/db timings; the model generation with tokens and latency | `grounded` (1/0) |
| `dev-team` (team run id as session) | `agent.planner`, `agent.coder`, `agent.reviewer`; a `tool.<name>` span per tool call; each planner generation | `review_approved` (1/0) |
| `agent.developer` (single-agent `dev-agent`) | `tool.<name>` spans and generations | — |

Useful views in Langfuse: cost and latency per agent role, the share of `rag.query` traces
with `grounded = 0`, review rounds per team run, and tool calls that ended in a policy block
(logged at `WARNING` level).

### Safety properties

- Off by default; the hosted deployment sends nothing unless configured.
- Fail-safe: a missing SDK, missing keys or an unreachable server degrade to a no-op tracer
  with a log warning; exceptions from traced code always propagate unchanged.
- Payloads are clipped (4,000 characters per field, 50 list items) before export.
- The RAG trace records the guardrail-sanitized query, not the raw input.
- Self-hosting keeps prompts, documents and outputs on your own infrastructure.
