# Grounded Agent Platform

**An applied-AI backend for answers and actions you can trust.** A multi-tenant RAG API that answers only from documents a user is allowed to see, and a developer agent whose tool use is controlled by policy outside the model. Both are measured by evaluation gates in CI.

[![CI](https://github.com/FarnazNK/grounded-agent-platform/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FarnazNK/grounded-agent-platform/actions/workflows/ci.yml)
[![Security](https://github.com/FarnazNK/grounded-agent-platform/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/FarnazNK/grounded-agent-platform/actions/workflows/security.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)

**[🖥️ Live demo](https://grounded-agent-web.vercel.app)** · **[📘 Live API docs](https://grounded-agent-api.vercel.app/docs)** · **[💚 Health](https://grounded-agent-api.vercel.app/health/ready)** · **[🧠 Core code](src/rag_agent/)** · **[📊 Evaluations](scripts/)**

> **Two live pieces.** The [web demo](https://grounded-agent-web.vercel.app) (React + TypeScript, in [`frontend/`](frontend/)) traces every question step by step: guardrails, dense and lexical search, rank fusion, grounding and citation checks. It runs a browser port of the backend's deterministic pipeline, so it needs no sign-in and makes no API calls. The [hosted API](https://grounded-agent-api.vercel.app/docs) is the real Python backend.

---

## At a glance

| | |
| --- | --- |
| **What it is** | (1) A RAG API: upload documents per workspace, ask questions, get answers with checked citations. (2) A developer agent that reads code, proposes changes and runs tests, within strict rules. |
| **AI techniques** | Hybrid retrieval (pgvector + full-text) with reciprocal-rank fusion, citation validation, input/context/output guardrails, tool-calling agent loop, reusable skills |
| **Backend** | Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL 16 + pgvector, Alembic, JWT |
| **Frontend** | React 18, TypeScript, Vite: a public demo that visualizes each pipeline step |
| **Models** | Anthropic or OpenAI adapters, plus deterministic offline providers for reproducible CI |
| **Quality** | 101 unit tests + PostgreSQL integration tests, RAG and adversarial eval gates, agent context eval, p95 latency gate, mypy, CodeQL, pip-audit |
| **Deployed on** | Vercel: RAG API with Neon PostgreSQL/pgvector, plus the static web demo |

## What I built

### 🔎 Hybrid retrieval with checked citations
Dense search (pgvector) finds meaning; full-text search catches exact terms like policy names, acronyms and IDs. The two rankings are merged with **weighted reciprocal-rank fusion**. Before an answer goes out, its citations are **validated against the retrieved sources**, and PII filters run on the output. Retrieval and embedding caches invalidate automatically when a document is re-indexed.
→ [`src/rag_agent/retrieval.py`](src/rag_agent/retrieval.py), [`src/rag_agent/service.py`](src/rag_agent/service.py)

### 🏢 Multi-tenant by design
Organizations → workspaces → viewer/editor/admin memberships. Every upload, re-index, delete and query is **scoped to a workspace the user belongs to**. The app refuses to start in staging or production if the admin-bootstrap shortcut is switched on.
→ [`src/rag_agent/auth.py`](src/rag_agent/auth.py)

### 🤖 A developer agent where *the model proposes and policy decides*
The agent picks the most relevant files within a fixed context budget, then plans with narrow tools (`list_files`, `read_file`, `search_code`, `write_file`, `run_check`, `finish`). Every tool call is checked **outside the model**: it can't leave the repo, can't touch credential-like paths, can only run allowlisted test/lint commands **without a shell**, and **can't write at all unless the operator passes `--allow-writes`**. Each run produces a step-by-step trace. Reusable YAML skills (`code-change`, `debug`, `review`) bring their own tool allowlists.
→ [`src/rag_agent/agent/`](src/rag_agent/agent/), [`agent_skills/`](agent_skills/)

### 📏 AI quality measured in CI, not assumed
Every pull request must pass:

| Gate | What it checks | Result on the CI sets |
| --- | --- | --- |
| RAG quality | Right source retrieved, answer grounded, citations correct | recall 1.0, citation correctness 1.0, hallucination 0.0 |
| Adversarial | Prompt-injection and unsafe requests get blocked | all blocked |
| Agent context | Agent selects the correct files for known tasks (top-8) | 1.0 (threshold 0.75) |
| Latency | p95 request latency on the benchmark | ~4 ms (limit 500 ms) |

Plus Ruff, mypy, unit and PostgreSQL/pgvector integration tests, `pip-audit`, a production Docker build, and **CodeQL**. Reports are uploaded as CI artifacts.
→ [`scripts/run_evals.py`](scripts/run_evals.py), [`data/eval_datasets/`](data/eval_datasets/), [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

### 🛡️ Layered guardrails, honestly labeled
Prompt-injection screening runs on user input **and** retrieved document text. It's regex-based, so it's treated as a first-pass filter that can be bypassed, **not** the security boundary. The real protection comes from authorization, tool policy and output validation.
→ [`src/rag_agent/guardrails/`](src/rag_agent/guardrails/)

## Architecture

```mermaid
flowchart LR
    subgraph RAG["RAG API · hosted"]
        C["Client"] --> A["FastAPI<br/>JWT + workspace auth"]
        A --> GI["Input guardrails"]
        GI --> R["Dense (pgvector) +<br/>full-text retrieval"]
        R --> F["Weighted RRF"]
        F --> GC["Context guardrails"]
        GC --> L["LLM<br/>Anthropic / OpenAI / offline"]
        L --> GO["Citation + output<br/>validation"]
    end
    DB[("PostgreSQL<br/>+ pgvector")]
    R --- DB

    subgraph DEV["Developer agent · local only"]
        T["Task"] --> CX["Context selection<br/>(budgeted)"]
        CX --> P["Tool planner"]
        P --> PO{"Policy check"}
        PO -->|allowed| TO["Tools: read · search ·<br/>write* · run_check"]
        TO --> TR["Run trace"]
    end

    E["Eval gates in CI"] -.-> R
    E -.-> CX
```

\* writes only with `--allow-writes`. The repo-writing agent is never exposed on the hosted API.

## Try it

**Hosted:** open the **[interactive API docs](https://grounded-agent-api.vercel.app/docs)**. Main routes: `POST /v1/documents/upload`, `POST /v1/query`, `GET /v1/workspaces`, plus `/health/live`, `/health/ready` and `/metrics`.

**Locally** (Python 3.11+, PostgreSQL with pgvector)
```bash
make dev                 # install with dev extras
cp .env.example .env     # offline providers by default, no API key needed
make migrate             # create tables
make test                # unit tests
make eval-ci             # RAG + adversarial evaluation gates
```

**Run the developer agent** (read-only by default)
```bash
rag-agent dev-agent --repo . --task "Trace how request authentication is enforced"

# with a real model and a skill
APP_LLM_PROVIDER=anthropic APP_LLM_MODEL=<model> \
  rag-agent dev-agent --repo . --skill review --task "Review the auth path for security issues"
```

## Honest limitations

- **Small CI eval sets:** the CI gates use small regression sets (3 RAG, 3 adversarial, 4 agent cases). Larger HR-policy sets (7 and 23 questions) run, but don't pass the answer-relevance gate with the offline providers yet, so they aren't in CI. On the 23-question set: recall 0.96, citation correctness 0.65.
- **Offline providers in the demo:** the hosted API uses deterministic embeddings and generation, so it shows the pipeline, not model-quality answers.
- **Agent sandboxing:** the agent's policy layer is defense in depth, not an OS sandbox. A hosted, write-enabled version would need ephemeral containers, network isolation and scoped credentials.

More detail: [`docs/limitations.md`](docs/limitations.md) and the architecture decision records in [`docs/adr/`](docs/adr/).

<details>
<summary><strong>Optional extras: AWS Bedrock/AgentCore, gRPC, Kubernetes, Prometheus/Grafana, Terraform</strong></summary>

None of these are deployed or needed to evaluate the project. All run locally without paid accounts.

- **AWS Bedrock + Strands + AgentCore:** a disabled-by-default, read-only, JWT/workspace-scoped agent reference with Bedrock Guardrails config. Run `python aws_agent/demo_offline.py`. Threat model and limitations in [`aws_agent/PORTFOLIO.md`](aws_agent/PORTFOLIO.md). Not deployed to AWS.
- **gRPC retrieval contract:** a Protocol Buffers service with deadlines and bearer metadata, on localhost only. Run `python -m grpc_demo.demo_offline`. See [`grpc_demo/retrieval.proto`](grpc_demo/retrieval.proto).
- **Kubernetes:** kind manifests with resource limits and secret references, validated in CI. See [`k8s/README.md`](k8s/README.md).
- **Prometheus + Grafana:** metrics for request latency, retrieval, model calls, tokens, ingestion and cache hits. See [`k8s/MONITORING.md`](k8s/MONITORING.md) and [`monitoring/`](monitoring/).
- **Terraform + AWS SAM/Lambda:** deployment scaffolding in [`infra/`](infra/), with a GitHub OIDC workflow.

</details>

## Repository guide

| Path | What's there |
| --- | --- |
| [`src/rag_agent/`](src/rag_agent/) | Retrieval, RAG service, auth, guardrails, evals, FastAPI app |
| [`src/rag_agent/agent/`](src/rag_agent/agent/) | Developer agent: context, planner, tools, policy, skills, harness |
| [`frontend/`](frontend/) | React + TypeScript web demo that traces the RAG pipeline |
| [`data/eval_datasets/`](data/eval_datasets/) | RAG quality and adversarial evaluation cases |
| [`tests/`](tests/) | Unit and PostgreSQL integration tests |
| [`docs/`](docs/) | Architecture, ADRs, developer-agent design, deployment, limitations |

---

Built by **Farnaz Nasehi** · [LinkedIn](https://www.linkedin.com/in/farnaz-nasehi/) · [GitHub](https://github.com/FarnazNK)
