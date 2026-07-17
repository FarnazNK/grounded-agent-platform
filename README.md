# RAG Agent

[![ci](https://github.com/FarnazNK/rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/FarnazNK/rag-agent/actions)
[![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![tests](https://img.shields.io/badge/tests-90%20passing-brightgreen.svg)](#testing)


A production-style RAG agent built on LangGraph with hybrid retrieval, an
evaluation harness, regex guardrails, a FastAPI service, Prometheus metrics,
and a comparative benchmark suite.

Built as a portfolio project for AI engineering / ML platform roles. The
emphasis is on the surfaces a real production system needs — eval, observability,
deployment, guardrails — not just on getting an LLM to answer a question.

> **Note:** This is a personal portfolio project. The HR/IT/security corpus is
> synthetic and unrelated to any real organization, employer, or client.

---

## Table of contents

- [What's in the box](#whats-in-the-box)
- [Serving layer: async, batching, backpressure](#serving-layer-async-batching-backpressure)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Benchmark results](#benchmark-results)
- [API reference](#api-reference)
- [Design decisions worth pointing at](#design-decisions-worth-pointing-at)
- [How this maps to AI engineering roles](#how-this-maps-to-ai-engineering-roles)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project structure](#project-structure)
- [What I'd build next](#what-id-build-next)

---

## Serving layer: async, batching, backpressure

The agent above answers questions. This layer is about serving it under
concurrent load, where the interesting problems are queueing and tail latency
rather than prompt quality.

### Fully async execution path

Every graph node is `async def`; LLM calls await the provider's native async
client, and the CPU-bound BM25 scan plus the blocking embeddings call are
offloaded with `asyncio.to_thread`. `/query` awaits `Agent.arun()`.

This was a real bug, not a refactor. The previous `/query` called the
synchronous `agent.run()` from an async handler, which blocks the event loop
for the entire multi-second run and serializes every concurrent request behind
it. Measured through the real HTTP stack with a mocked LLM:

| Concurrency | Serialized (blocking) | Async | Speedup |
|---:|---:|---:|---:|
| 8 | ~1600 ms | 248 ms | 6.4x |
| 32 | ~6400 ms | 344 ms | 18.6x |

`tests/test_async_path.py` locks this in with a concurrency test that fails if
anyone reintroduces a blocking call, plus a control test proving the probe can
actually detect a stall.

`arun` takes a deadline and is cancellable — cancelling the task unwinds the
graph at whatever await is in flight, which is the mechanism barge-in needs.

### Dynamic batch scheduler

`src/rag_agent/voice/scheduler.py` — a bounded queue in front of a batching
consumer:

- A batch closes when it hits `max_batch_size` **or** `max_wait_ms` elapses
  since the batch's *first* request. Anchoring the deadline to the first
  request (not the last) is what prevents a steady arrival stream from
  postponing the batch forever and starving the oldest caller.
- The queue is bounded. At capacity, `submit()` raises `QueueFullError`
  instead of buffering — an unbounded queue converts overload into an OOM,
  and work whose caller already hung up is pure waste. Shedding load fast is
  the feature.
- Requests carry deadlines; expired ones are dropped before inference rather
  than spending device time on a result nobody will read.
- One bad batch fails only its own callers; the consumer loop survives.

### Measured results

Full reports in [`benchmarks/load/results/`](benchmarks/load/results/), with
the harness in [`benchmarks/load/`](benchmarks/load/README.md).

> **These are scheduler numbers, not speech-model numbers.** The default
> backend is `SimulatedSTT`, which sleeps according to an explicit cost model
> and transcribes nothing. It exists so the serving path is testable without a
> GPU. Swap in a faster-whisper provider on a GPU host and re-run the same
> harness for real figures — the interface, sweep, and metrics don't change.

**`max_wait_ms=0` silently disables batching.** Mean batch size stays at 1.00
for *every* batch cap from 1 to 16; throughput pins at ~28 rps. The cap alone
does nothing — batching needs a window to accumulate against. This is exactly
what the `voice_inference_batch_size` histogram exists to catch in production.

**Above saturation, batching improves latency *and* throughput.** At
concurrency 32 with a 5 ms window, batch 1 → 16 gave ~7.6x throughput
(28 → 216 rps) and ~7.6x lower p50 (1132 → 149 ms). The textbook
latency-for-throughput tradeoff holds at concurrency 1, where a lone request
pays the full window for nothing — and inverts under load, where queue wait
dominates and draining faster wins on both axes.

**Clean saturation knee at ~8 concurrent clients** (2 s segments):

| Concurrency | p50 | p95 | p99 | RPS | Queue p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 45.8 ms | 48.9 ms | 50.3 ms | 21.9 | 10.5 ms |
| 8 | 54.9 ms | 60.8 ms | 62.9 ms | 144.1 | 0.2 ms |
| 32 | 221.5 ms | 252.2 ms | 257.2 ms | 142.7 | 196.4 ms |
| 64 | 438.7 ms | 449.5 ms | 455.3 ms | 145.5 | 394.6 ms |

Throughput plateaus at ~145 rps from concurrency 8 while latency grows
linearly. At 64, queue wait is 395 ms of the 439 ms total — 90% of latency is
pure queueing. Past the knee the answer is capacity, not tuning.

Percentiles use nearest-rank (not interpolation), so every figure is a real
observation; p99 below 100 samples is flagged rather than quoted.

### Stage-level metrics

`voice_stage_latency_seconds{stage=stt|agent|tts}`,
`voice_inference_batch_size`, `voice_inference_queue_wait_seconds`,
`voice_audio_queue_depth`, `voice_transcription_realtime_factor`,
`voice_dropped_audio_chunks_total`, and others. Request-level latency can't
answer the only question that matters during a p99 regression — *which
stage?* — so the histograms are labelled by stage and bucketed for the
100 ms–2 s range voice SLOs actually live in.

### What is not built

Streaming STT/TTS with real models, WebSocket audio sessions, barge-in, GPU
profiling, and rollout controls are **not implemented**. They need a GPU and
real model weights; writing that code without being able to run it would mean
publishing latency claims I haven't measured. The provider interface
(`SpeechToTextProvider` / `TextToSpeechProvider`) and the cancellation
plumbing are in place so those land without touching orchestration.

---

## What's in the box

Six things, wired together:

1. **An agent** — a LangGraph DAG that routes queries, rewrites them for
   better retrieval, grades the retrieved context, and loops if the grade
   is poor. Topology, not a ReAct loop, so control flow is explicit and
   debuggable.
2. **A hybrid RAG pipeline** — dense (Chroma) + sparse (BM25) retrieval
   fused with reciprocal rank fusion. The vector store sits behind a
   `Protocol`, so swapping Chroma for Pinecone is one class.
3. **An eval harness** — YAML-defined test cases, four pluggable scorers
   (exact match, substring, retrieval recall, LLM-as-judge), a runner
   with Rich console output, JSON export, and a `--fail-under` threshold
   that makes it CI-ready.
4. **Guardrails** — regex-based PII detection (sanitize or block), prompt
   injection detection, output PII-leak detection, and source citation
   verification. Pluggable via the same `Protocol` pattern as scorers.
5. **A FastAPI service** — `POST /query`, `POST /stream` (Server-Sent
   Events), `GET /health`, `GET /metrics` (Prometheus). Guardrails wired
   into the request path.
6. **A comparative benchmark suite** — runs the eval dataset under multiple
   agent configurations side by side and produces a Markdown comparison
   table. This is what an eval framework is *for*: comparable numbers
   across changes, not one number in isolation.

A 60-document sample corpus (HR, IT, security, finance, engineering policies)
and a 23-case eval suite ship with the repo so the whole thing runs end to
end with one command.

---

## Quickstart

```bash
# Clone + install
pip install -e ".[dev,api,tracing]"

# Set API keys
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (for generation) and OPENAI_API_KEY (for embeddings)

# Generate the extended corpus (60 docs)
make corpus

# Run the API
make serve
# → http://localhost:8000/docs (interactive OpenAPI UI)

# Or run via Docker
make docker-up
```

Once the API is up:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many PTO days do new hires get?"}'
```

```json
{
  "answer": "New hires accrue 15 PTO days per year for the first 2 years. [source: pto_policy.md]",
  "route": "retrieve",
  "iterations": 1,
  "chunks": [{"source": "pto_policy.md", "score": 0.92, "snippet": "..."}],
  "latency_ms": 1247.3,
  "run_id": "f7c2...",
  "guardrails": [
    {"name": "pii_detector", "action": "allow", "reason": "no pii detected"},
    {"name": "prompt_injection_detector", "action": "allow", "reason": "no injection detected"},
    {"name": "pii_leak_detector", "action": "allow", "reason": "no pii in output"}
  ]
}
```

CLI usage works too:

```bash
make ingest                                          # index the corpus
make ask Q='What does it take to get promoted?'      # ask, with --trace
make eval-fast                                       # run the eval suite (no LLM judge)
make benchmark                                       # run all configs side by side
```

---

## Architecture

```
                         ┌──────────────┐
                  HTTP   │   FastAPI    │   /query  /stream  /health  /metrics
                  ────►  │   service    │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  Guardrails  │   PII / prompt-injection / source-cite
                         │   pipeline   │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                  START ►│    route     │
                         └──────┬───────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
         refuse          answer_directly         retrieve
            │                   │                   │
            ▼                   ▼                   ▼
       ┌─────────┐        ┌──────────┐       ┌──────────────┐
       │ refuse  │        │  direct  │       │ rewrite_query│
       │  node   │        │  answer  │       └──────┬───────┘
       └────┬────┘        └────┬─────┘              │
            │                  │                    ▼
            │                  │              ┌──────────┐
            │                  │              │ retrieve │  (dense + BM25 + RRF)
            │                  │              └─────┬────┘
            │                  │                    │
            │                  │                    ▼
            │                  │              ┌──────────┐
            │                  │              │  grade   │
            │                  │              └─────┬────┘
            │                  │                    │
            │                  │      ┌─────────────┴────────────┐
            │                  │   relevant                irrelevant + iter<N
            │                  │      │                          │
            │                  │      ▼                          │
            │                  │ ┌──────────┐                    │
            │                  │ │ generate │◄── irrelevant + iter≥N
            │                  │ └────┬─────┘
            │                  │      │
            └──────────────────┴──────┴──────► output guardrails ──► response
```

The graph state (`AgentState`) is a Pydantic model that flows through every
node. Each node returns a partial-state dict; LangGraph merges the updates.

---

## Benchmark results

The point of an eval framework isn't to produce a single number — it's to
produce *comparable* numbers across configurations. The benchmark runs the
same 23-case dataset under five agent configurations and writes a comparison.

Run it yourself with `make benchmark`. Results below are from a representative
run on the 60-document HR/IT corpus (Claude Sonnet 4.5, OpenAI
text-embedding-3-small).

### Pass rate by config

| Config              | Pass rate | Notes                                                                |
|---------------------|----------:|----------------------------------------------------------------------|
| `baseline`          |       91% | All features on — hybrid retrieval, query rewrite, grading loop.     |
| `dense_only`        |       74% | BM25 disabled. Acronym queries (PIP, VPN, HSA, BYOD) regress sharply.|
| `no_query_rewrite`  |       83% | Skipping rewrite hurts paraphrase queries the most.                  |
| `no_grading`        |       78% | Without the grader, irrelevant first-shot retrieval is committed to. |
| `single_iter`       |       83% | Loop disabled; some recoverable failures become hard failures.       |

### Key findings

- **Hybrid retrieval is worth it.** The drop from `baseline` to `dense_only`
  is concentrated in queries with acronyms ("How long is a PIP?", "What VPN
  do we use?"). Pure dense embeddings semantically match documents that
  *talk about* PIP without containing the term. BM25 catches the literal
  keyword.
- **Query rewriting matters most for paraphrases.** "What's our company
  stance on unnecessary meetings?" is far from "meeting agenda required" in
  embedding space. The rewriter bridges the gap.
- **The grader earns its latency.** Without it, ~15% of cases that the loop
  could have recovered from become committed-to wrong answers.

The full per-case breakdown lands in `benchmarks/results/comparison.md`
after each run. The JSON output (`benchmarks/results/{config}.json`) is the
machine-readable version used by CI.

---

## API reference

OpenAPI / Swagger UI is auto-generated and served at `/docs`.

### `POST /query`

Synchronous one-shot RAG response. Input guardrails sanitize PII; prompt
injection attempts return 400. Output guardrails sanitize PII leaks and
unverified citations.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How long is parental leave?"}'
```

Response: `QueryResponse` (see `src/rag_agent/api/schemas.py`).

### `POST /stream`

Server-Sent Events stream. Useful for chat UIs that want to render tokens
as they arrive.

```bash
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "How does code review work here?"}'
```

Event types:
- `event: node` — graph node lifecycle (`start` / `end`)
- `event: token` — streaming LLM tokens
- `event: done` — final answer + metadata
- `event: error` — terminal error

### `GET /health`

Liveness check. Reports corpus size, vector count, and whether tracing is on.

### `GET /metrics`

Prometheus exposition. Counters: `rag_agent_requests_total`,
`rag_agent_guardrail_blocks_total`, `rag_agent_guardrail_sanitizations_total`.
Histograms: `rag_agent_request_latency_seconds`, `rag_agent_loop_iterations`.

---

## Design decisions worth pointing at

A few places where I made a non-obvious choice and the reasoning behind it:

**Topology, not ReAct.** A ReAct-style "let the LLM decide what to do next"
loop is seductive but hard to evaluate — every failure mode looks the same.
An explicit graph makes the router, rewriter, and grader addressable as
separate components. You can measure router accuracy independently of
generation quality. You can swap the grader without touching the retriever.

**Hybrid retrieval, not pure dense.** Vector search is great for semantic
recall but fragile for acronyms, IDs, and exact product names — all of which
show up constantly in HR / enterprise queries ("what about PTO?", "find me
the PIP policy"). BM25 on the same corpus catches those. RRF is the cheapest
robust fusion method; the `k=60` constant is the default from Cormack et al.

**Grader with a fail-open policy.** If the grader's JSON output is malformed,
we treat the chunks as relevant rather than blocking the user. A broken grader
should not cause user-visible failures. It does get logged, loudly.

**Regex guardrails before model-based ones.** Three reasons: latency (sub-ms),
predictability (the eval suite needs deterministic behavior), and
auditability (a regex is reviewable; an LLM judge isn't). Model-based
guardrails belong on top of these, not instead of them.

**Sanitize, don't block, by default.** Most enterprise users want their
request to proceed with PII redacted, not be hard-blocked. Block only on
genuine integrity threats (prompt injection, intentional data exfiltration
attempts). Each guardrail has a `mode` knob.

**LangSmith is optional.** If `LANGSMITH_API_KEY` is set, traces flow there
automatically via LangChain's auto-tracing. If not, structured logs work
identically. Observability shouldn't be a hard dependency on a vendor.

**Lazy `Agent` import.** `from rag_agent import Agent` works, but the
top-level `__init__.py` uses PEP 562 lazy loading so importing
`rag_agent.schemas` doesn't drag in LangGraph. This matters for deployment
scenarios where the eval scorers run in a smaller environment.

**Scorers and guardrails are Protocols, not base classes.** Any callable
with the right signature qualifies. Adding a new metric or guardrail is a
function, not a subclass.

**SSE, not WebSocket, for streaming.** SSE is simpler, has built-in
reconnection, and works through every proxy. The trade-off is one-way
communication, which is fine for a Q&A agent. WebSocket would be the right
call if I needed bidirectional flow (e.g. interrupting generation).

---

## How this maps to AI engineering roles

Common requirements for AI engineering / ML platform roles and where each
one lives in the code:

| Requirement                                | Where it lives                                          |
|--------------------------------------------|---------------------------------------------------------|
| LangGraph agent orchestration              | `src/rag_agent/graph.py`                                |
| RAG pipelines + retrieval optimization     | `src/rag_agent/retrieval.py` (hybrid + RRF)             |
| Vector database                            | `src/rag_agent/vectorstore.py` (Chroma, swappable)      |
| Evaluation framework (offline evals)       | `src/rag_agent/evals/`                                  |
| Evaluation datasets + test harnesses       | `data/eval_datasets/hr_full.yaml`, `evals/runner.py`    |
| Automated scoring pipelines                | `scripts/run_evals.py` with `--fail-under` for CI       |
| Hallucination / response quality metrics   | `LLMJudgeScorer`, `RetrievalRecallScorer`               |
| Comparative benchmarking                   | `benchmarks/compare_configs.py`                         |
| Guardrails / PII / prompt injection        | `src/rag_agent/guardrails/`                             |
| Production API + streaming                 | `src/rag_agent/api/` (FastAPI, SSE)                     |
| Observability — logs, traces, metrics      | `src/rag_agent/observability.py`, `api/metrics.py`      |
| Build-vs-buy abstractions                  | `VectorStore` Protocol, `build_llm` factory             |
| Production-grade Python                    | Pydantic schemas, tenacity retries, structlog, tests    |
| Containerization + CI                      | `Dockerfile`, `docker-compose.yml`, `.github/workflows/`|

---

## Testing

50 unit tests run with no API keys and no network access — LLMs are stubbed
at the module boundary so the entire graph and API can run in CI deterministically.

```bash
make test         # all tests with full output
make test-fast    # only the no-network suites
```

Coverage by module:

| Module                  | Test file                  | Coverage |
|-------------------------|----------------------------|----------|
| State schemas           | `test_schemas.py`          | Contract validation |
| Hybrid retrieval / RRF  | `test_retrieval.py`        | Determinism, fusion math |
| Eval scorers            | `test_scorers.py`          | All four scorer types |
| LangGraph topology      | `test_graph.py`            | All three paths (retrieve / direct / refuse) + loop cap |
| Guardrails              | `test_guardrails.py`       | All filters + pipeline composition |
| FastAPI surface         | `test_api.py`              | Health, query, metrics, guardrail wiring |

CI runs lint (ruff), format check, type check (mypy), tests, and a Docker
build smoke test on every PR. See `.github/workflows/ci.yml`.

---

## Deployment

### Docker

The Dockerfile is multi-stage (builder + runtime), runs as a non-root user,
and ships with a Python-based healthcheck so orchestrators can detect
wedged processes. The compose file adds a persistent Chroma volume and a
one-shot `evals` service.

```bash
make docker-up           # start the API stack
make docker-down         # stop and remove
docker compose run --rm evals   # run the eval suite against the live API
```

### Environment variables

All config flows through pydantic-settings. The most relevant:

| Variable                  | Default              | Purpose                          |
|---------------------------|----------------------|----------------------------------|
| `ANTHROPIC_API_KEY`       | (required)           | Generation                       |
| `OPENAI_API_KEY`          | (required)           | Embeddings                       |
| `APP_LLM_MODEL`           | `claude-sonnet-4-5`  | Generation model                 |
| `APP_TOP_K_DENSE`         | `8`                  | Dense candidates per query       |
| `APP_TOP_K_SPARSE`        | `8`                  | BM25 candidates per query        |
| `APP_TOP_K_FINAL`         | `4`                  | Chunks passed to the LLM         |
| `APP_MAX_ITERATIONS`      | `3`                  | Retrieve-grade loop cap          |
| `APP_ENABLE_QUERY_REWRITE`| `true`               | Toggle the rewriter              |
| `APP_ENABLE_GRADING`      | `true`               | Toggle the grader                |
| `LANGSMITH_API_KEY`       | (unset)              | Enable LangSmith tracing         |

### Production considerations not handled

- **Authentication.** The API has no auth layer. Add a JWT/API-key middleware
  at the FastAPI level before exposing it externally.
- **Rate limiting.** Per-user / per-IP quotas. `slowapi` is the standard
  choice. Wire into the same middleware as auth.
- **Concurrency.** A single uvicorn worker is fine for a portfolio demo; a
  real deployment wants multiple workers behind a load balancer, with
  Chroma replaced by a shared backend (pgvector, Pinecone) so workers don't
  fight over the local store.
- **Secrets.** API keys are read from env vars. In production, fetch from a
  secret manager (Vault, AWS Secrets Manager, GCP Secret Manager) on startup.

---

## Project structure

```
rag-agent/
├── src/rag_agent/
│   ├── __init__.py            # public API surface, PEP 562 lazy imports
│   ├── agent.py               # public Agent class (sync + async stream)
│   ├── config.py              # pydantic-settings
│   ├── graph.py               # LangGraph topology
│   ├── llm.py                 # LLM provider factory + retries
│   ├── observability.py       # structlog + optional LangSmith
│   ├── prompts.py             # centralized prompt templates
│   ├── retrieval.py           # hybrid dense + BM25 + RRF
│   ├── schemas.py             # Pydantic state + chunk types
│   ├── vectorstore.py         # Chroma wrapper behind a Protocol
│   ├── cli.py                 # typer CLI
│   ├── api/
│   │   ├── app.py             # FastAPI factory + routes
│   │   ├── schemas.py         # HTTP request/response models
│   │   └── metrics.py         # Prometheus collectors
│   ├── guardrails/
│   │   ├── base.py            # Guardrail Protocol + apply_guardrails
│   │   ├── input_filters.py   # PII / prompt injection / profanity
│   │   └── output_filters.py  # PII leak / source citation
│   └── evals/
│       ├── dataset.py         # YAML loader + EvalCase / EvalDataset
│       ├── scorers.py         # ExactMatch / Contains / RetrievalRecall / LLMJudge
│       └── runner.py          # runs cases, aggregates, prints report
├── tests/                     # 50 tests; no API keys required
├── scripts/
│   ├── generate_corpus.py     # build the 60-doc HR/IT/security corpus
│   └── run_evals.py           # CLI eval runner with --fail-under
├── benchmarks/
│   └── compare_configs.py     # comparative benchmark across agent configs
├── data/
│   ├── sample_corpus_extended/    # 60 markdown docs
│   └── eval_datasets/             # YAML test cases
├── .github/workflows/ci.yml   # lint, typecheck, tests, docker build
├── Dockerfile                 # multi-stage, non-root, healthcheck
├── docker-compose.yml         # api + evals services + persistent Chroma
├── Makefile                   # one-line entry points
├── pyproject.toml             # deps, ruff, mypy, pytest config
└── README.md
```

---

## What I'd build next

The three things I'd prioritize if this became a real production project:

1. **Authentication + rate limiting.** A JWT middleware and per-user quotas
   are table stakes for any external-facing AI service. The pattern is well
   understood; it's just work I haven't done here.
2. **Streaming citations.** The answer prompt asks for `[source: file]`
   citations, but a real system should track which specific chunks
   supported which claims at the token level. Useful for hallucination
   detection in eval and for clickable source links in the UI.
3. **Evaluation dashboard.** The JSON output from `run_evals.py` and the
   benchmark results are structured, but reading them across many runs is
   tedious. A small Streamlit app that compares runs over time would make
   regressions much easier to spot.

A few things I deliberately *didn't* build and would flag up front in an
interview:

- **No fine-tuning pipeline.** Some roles list RLHF / LoRA as nice-to-haves.
  I chose breadth (eval + agent + RAG + API + guardrails) over depth in any
  one area. Fine-tuning without an eval framework to measure its effect is
  the wrong order anyway.
- **No multi-agent topology.** Some roles call for it. I think multi-agent
  systems are usually the wrong abstraction when a well-designed
  single-agent graph would do, and wanted this project to reflect that
  opinion.


---

*Built by Farnaz Nasehi.*
