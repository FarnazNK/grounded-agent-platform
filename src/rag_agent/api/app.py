"""FastAPI application factory.

Endpoints:
- POST /query   — synchronous one-shot RAG response
- POST /stream  — Server-Sent Events stream of node + token events
- GET /health   — liveness/readiness with corpus stats
- GET /metrics  — Prometheus exposition

Design notes:
- The Agent is built once at startup and reused across requests. Constructing
  it per-request would cost us a Chroma reconnect every time.
- Guardrails run synchronously in the request path. They're regex-based and
  measured at ~0.17 ms for a 440-char query, so offloading them to a thread
  would cost more in hop overhead than it saves. Model-based guardrails would
  need to move off the loop (or to a sidecar).
- Every route is async and every call inside it either awaits or is measured
  fast enough to run inline. Nothing calls `agent.run()` — see `arun`.
- SSE (`/stream`) stays for text clients: simple, auto-reconnecting, proxy-
  friendly, and one-way is fine for Q&A. Voice needs the client to send audio
  while the server streams back, so it gets a WebSocket endpoint instead.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.documents import Document

from rag_agent.agent import Agent
from rag_agent.api.metrics import (
    GUARDRAIL_BLOCKS,
    GUARDRAIL_SANITIZATIONS,
    ITERATIONS_OBSERVED,
    REQUEST_LATENCY,
    REQUESTS,
    render_metrics,
)
from rag_agent.api.schemas import (
    ChunkOut,
    GuardrailEventOut,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from rag_agent.config import get_settings
from rag_agent.guardrails import (
    PIIDetector,
    PIILeakDetector,
    PromptInjectionDetector,
    apply_guardrails,
)
from rag_agent.guardrails.base import (
    GuardrailAction,
    GuardrailDecision,
    GuardrailViolation,
)
from rag_agent.observability import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: build the agent once
# ---------------------------------------------------------------------------


def _load_corpus(data_dir: Path) -> list[Document]:
    """Load every .md file in `data_dir` as one document. One file = one doc.

    For real deployments this would do proper chunking; for the portfolio
    project the docs are small enough that file-as-chunk works fine.
    """
    if not data_dir.exists():
        return []
    docs: list[Document] = []
    for i, path in enumerate(sorted(data_dir.glob("*.md"))):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name, "chunk_id": f"doc-{i}"},
            )
        )
    return docs


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the agent on startup, store on app.state."""
    data_dir = Path(app.state.data_dir)
    docs = _load_corpus(data_dir)
    if not docs:
        log.warning("api.startup.no_corpus", data_dir=str(data_dir))
        # Still build an empty agent so /health works and clients get a clear
        # error from /query rather than a startup crash.
    app.state.agent = Agent.from_corpus(docs) if docs else None
    app.state.corpus_size = len(docs)
    app.state.input_guardrails = [PIIDetector(), PromptInjectionDetector()]
    app.state.output_guardrails = [PIILeakDetector()]
    log.info("api.ready", corpus_size=len(docs))
    yield
    log.info("api.shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(data_dir: Path | str | None = None) -> FastAPI:
    """Build the FastAPI app. `data_dir` overrides the default corpus path."""
    settings = get_settings()
    app = FastAPI(
        title="RAG Agent API",
        description="LangGraph RAG agent with hybrid retrieval, guardrails, and evals.",
        version="0.2.0",
        lifespan=_lifespan,
    )
    app.state.data_dir = str(
        data_dir or Path(__file__).resolve().parents[3] / "data" / "sample_corpus_extended"
    )
    app.state.settings = settings

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Cheap liveness check. Always returns 200 if the process is up.

        Reports vector_count separately from corpus_size so we can detect
        the case where the agent built but the vector store is empty
        (e.g. embeddings call failed silently).
        """
        agent: Agent | None = app.state.agent
        return HealthResponse(
            status="ok" if agent is not None else "degraded",
            corpus_size=app.state.corpus_size,
            vector_count=agent._retriever.store.count() if agent else 0,
            tracing_enabled=bool(agent and agent._tracing_enabled),
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus exposition endpoint. Plain text, no JSON wrapping."""
        return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest, request: Request) -> Response:
        agent: Agent | None = app.state.agent
        if agent is None:
            raise HTTPException(503, detail="Agent not initialized — corpus is empty.")

        endpoint = "query"
        start = time.perf_counter()

        # --- input guardrails ------------------------------------------------
        try:
            sanitized_query, in_decisions = apply_guardrails(req.query, app.state.input_guardrails)
        except GuardrailViolation as exc:
            GUARDRAIL_BLOCKS.labels(guardrail=exc.decision.guardrail_name).inc()
            REQUESTS.labels(endpoint=endpoint, status="blocked").inc()
            return JSONResponse(
                status_code=400,
                content={
                    "error": "request blocked by guardrail",
                    "guardrail": exc.decision.guardrail_name,
                    "detail": exc.decision.reason,
                },
            )
        _record_sanitizations(in_decisions)

        # --- run the agent ---------------------------------------------------
        # `arun`, not `run`: this is an async handler, so a synchronous graph
        # call here would block the single event loop thread for the entire
        # multi-second run and serialize every concurrent request behind it.
        try:
            state = await agent.arun(
                sanitized_query,
                timeout=app.state.settings.request_timeout_seconds,
            )
        except TimeoutError as exc:
            log.warning("api.query.timeout", query=sanitized_query[:60])
            REQUESTS.labels(endpoint=endpoint, status="timeout").inc()
            raise HTTPException(504, detail="agent run exceeded deadline") from exc
        except Exception as exc:
            log.exception("api.query.failed", error=str(exc))
            REQUESTS.labels(endpoint=endpoint, status="error").inc()
            raise HTTPException(500, detail=str(exc)) from exc

        ITERATIONS_OBSERVED.observe(state.iterations)

        # --- output guardrails ----------------------------------------------
        answer = state.final_answer or ""
        try:
            sanitized_answer, out_decisions = apply_guardrails(
                answer, app.state.output_guardrails, raise_on_block=False
            )
        except GuardrailViolation:
            # Output guardrails don't raise (raise_on_block=False above), but
            # belt-and-suspenders: if one ever does, return the unsanitized
            # answer rather than a 500. Worse to lose the user's response than
            # to log the violation and continue.
            sanitized_answer = answer
            out_decisions = []
        _record_sanitizations(out_decisions)

        latency_s = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_s)
        REQUESTS.labels(endpoint=endpoint, status="ok").inc()

        return JSONResponse(
            content=QueryResponse(
                answer=sanitized_answer,
                route=state.route,
                iterations=state.iterations,
                chunks=[
                    ChunkOut(
                        source=c.source,
                        score=c.score,
                        snippet=c.content[:200] + ("..." if len(c.content) > 200 else ""),
                    )
                    for c in state.chunks
                ],
                latency_ms=latency_s * 1000,
                run_id=state.run_id,
                guardrails=[
                    GuardrailEventOut(
                        name=d.guardrail_name,
                        action=d.action.value,
                        reason=d.reason,
                    )
                    for d in [*in_decisions, *out_decisions]
                ],
                sanitized_query=sanitized_query if sanitized_query != req.query else None,
            ).model_dump()
        )

    @app.post("/stream")
    async def stream(req: QueryRequest) -> StreamingResponse:
        """SSE stream of agent events.

        Each event is `event: <type>\\ndata: <json>\\n\\n`. Types:
        - `node`  : a graph node started or finished
        - `token` : a streaming LLM token
        - `done`  : final answer with metadata
        - `error` : terminal error
        """
        agent: Agent | None = app.state.agent
        if agent is None:
            raise HTTPException(503, detail="Agent not initialized.")

        # Apply input guardrails before starting the stream — once we've
        # opened the stream, error reporting is awkward. Catch up front.
        try:
            sanitized_query, _ = apply_guardrails(req.query, app.state.input_guardrails)
        except GuardrailViolation as exc:
            GUARDRAIL_BLOCKS.labels(guardrail=exc.decision.guardrail_name).inc()
            raise HTTPException(
                400,
                detail=f"blocked by {exc.decision.guardrail_name}: {exc.decision.reason}",
            ) from exc

        return StreamingResponse(
            _event_generator(agent, sanitized_query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    return app


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------


async def _event_generator(agent: Agent, query: str) -> AsyncIterator[bytes]:
    """Translate LangGraph's async events into SSE frames.

    LangGraph's `astream_events` emits a fine-grained event per node and per
    LLM token. We forward node-level lifecycle events and chat token events,
    and synthesize a `done` frame with the final state.
    """
    final_state: dict[str, Any] = {}

    try:
        async for event in agent.astream_events(query):
            kind = event.get("event")
            name = event.get("name", "")
            data = event.get("data", {})

            if kind == "on_chain_start" and name in {
                "route",
                "rewrite_query",
                "retrieve",
                "grade",
                "generate",
                "direct_answer",
                "refuse",
            }:
                yield _sse("node", {"name": name, "phase": "start"})

            elif kind == "on_chain_end" and name in {
                "route",
                "rewrite_query",
                "retrieve",
                "grade",
                "generate",
                "direct_answer",
                "refuse",
            }:
                yield _sse("node", {"name": name, "phase": "end"})

            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is None:
                    continue
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                if content:
                    yield _sse("token", {"text": content})

            elif kind == "on_chain_end" and name == "LangGraph":
                # Final state lands here.
                output = data.get("output", {})
                if isinstance(output, dict):
                    final_state = output

        yield _sse(
            "done",
            {
                "answer": final_state.get("final_answer") or "",
                "route": final_state.get("route"),
                "iterations": final_state.get("iterations", 0),
            },
        )
    except Exception as exc:
        log.exception("api.stream.failed", error=str(exc))
        yield _sse("error", {"message": str(exc)})


def _sse(event: str, data: dict) -> bytes:
    """Format a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


def _record_sanitizations(decisions: list[GuardrailDecision]) -> None:
    """Bump the sanitization counter for any guardrail that altered text."""
    for d in decisions:
        if d.action == GuardrailAction.SANITIZE:
            GUARDRAIL_SANITIZATIONS.labels(guardrail=d.guardrail_name).inc()
