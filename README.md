# RAG Agent

A measured RAG API for document ingestion, hybrid retrieval, tenant isolation, and observable AI serving.

## 1) What it does
- Authenticated API for document upload, re-index, query, and deletion.
- Persistent ingestion jobs with status tracking and clear failure codes.
- Hybrid retrieval using PostgreSQL full-text search plus pgvector dense search.
- Tenant-scoped organizations and workspaces enforced across ingestion and query paths.

## 2) Architecture
- FastAPI API
- PostgreSQL + pgvector persistence
- JWT authentication
- Structured logging + Prometheus metrics + Grafana dashboards
- Deterministic local providers by default; OpenAI/Anthropic adapters available for deployment

See `docs/architecture.md`.

## 3) Retrieval pipeline
1. Validate JWT and workspace membership.
2. Validate file/query input and prompt-injection patterns.
3. Parse and chunk documents.
4. Generate embeddings and persist chunks.
5. Retrieve dense and lexical candidates.
6. Fuse candidates with reciprocal-rank fusion.
7. Generate a cited answer and verify citations.

## 4) Evaluation
- Dataset: `data/eval_datasets/quality.yaml`
- Adversarial dataset: `data/eval_datasets/adversarial.yaml`
- Command: `make eval`
- CI gates:
  - retrieval recall >= 0.75
  - groundedness >= 0.90
  - p95 latency <= 500 ms

## 5) Reliability
- Timeouts and bounded retries for LLM/embedding providers.
- Explicit error taxonomy for invalid uploads, duplicate ingestion, malformed model output, token limits, rate limits, and database failures.
- Re-index and delete operations invalidate retrieval cache through workspace index versions.

## 6) Security
- JWT-protected APIs
- Workspace membership and role checks
- MIME and file-size validation
- Prompt-injection and PII guardrails
- Dependency scanning in CI (`pip-audit`)

## 7) Observability
- `X-Request-ID` correlation IDs
- JSON structured logs
- Prometheus metrics for HTTP, retrieval, LLM, DB, tokens, cache hits, and ingestion failures
- Grafana dashboard: `monitoring/grafana/dashboards/rag-agent-overview.json`

## 8) Deployment
- Local stack: `docker compose up --build`
- Migrations: `alembic upgrade head`
- Terraform: `infra/terraform`
- Deployment flow and rollback notes: `docs/deployment.md`

## 9) Benchmarks
- k6 suite: `benchmarks/k6/query.js`
- CI benchmark: `python benchmarks/load/run_benchmark.py --assert-p95-ms 500`
- CI uploads benchmark/evaluation reports as artifacts for each run

## 10) Design decisions
See `docs/adr/` for hybrid retrieval, pgvector, FastAPI, chunking, rank fusion, and retry boundaries.

## 11) Limitations
See `docs/limitations.md` for current limits, 10x/100x evolution plans, and non-goals.

## Quick start

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,tracing]'
cp .env.example .env
docker compose up -d postgres
alembic upgrade head
rag-agent bootstrap-demo
uvicorn rag_agent.api:create_app --factory --reload
```

Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,tracing]"
Copy-Item .env.example .env
docker compose up -d postgres
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\rag-agent.exe bootstrap-demo
.\.venv\Scripts\uvicorn.exe rag_agent.api:create_app --factory --reload
```

## API surface
- `POST /v1/auth/bootstrap`
- `POST /v1/auth/login`
- `GET /v1/workspaces`
- `POST /v1/documents/upload`
- `POST /v1/documents/{id}/reindex`
- `DELETE /v1/documents/{id}`
- `POST /v1/query`
- `GET /health/live`
- `GET /health/ready`
- `GET /metrics`
