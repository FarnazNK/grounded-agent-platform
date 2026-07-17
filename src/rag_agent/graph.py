"""LangGraph topology.

The graph is intentionally explicit — no auto-magic. Each node is a pure-ish
function from state to state-update, which makes them easy to test in
isolation and easy to swap.

Every node is `async def`. The nodes that call an LLM await the provider's
native async client; the retrieve node awaits a thread-offloaded hybrid
search. Nothing in a node body may block the event loop — under concurrent
voice sessions a single blocking call stalls every other session sharing the
worker, which shows up directly as p99 tail latency.

A uniformly async graph also means `ainvoke`/`astream_events` never silently
fall back to running a sync node in LangGraph's default thread pool, which
would cap effective concurrency at the pool size.

Topology:

    START
      │
      ▼
    [route] ──refuse──▶ [refuse_node] ──▶ END
      │
      ├──answer_directly──▶ [direct_answer] ──▶ END
      │
      └──retrieve──▶ [rewrite_query] ──▶ [retrieve] ──▶ [grade]
                                                          │
                              ┌───────irrelevant──────────┤
                              ▼                           ▼
                        (loop, max_iter)              [generate] ──▶ END
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from rag_agent.config import get_settings
from rag_agent.llm import build_llm, invoke_with_retry_async
from rag_agent.observability import get_logger
from rag_agent.prompts import (
    build_answer_prompt,
    build_grader_prompt,
    build_rewrite_prompt,
    build_router_prompt,
)
from rag_agent.retrieval import HybridRetriever
from rag_agent.schemas import AgentState, GradingResult, RouteDecision

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _make_route_node():
    llm = build_llm()

    async def route_node(state: AgentState) -> dict:
        messages = build_router_prompt(state.query)
        raw = (await invoke_with_retry_async(llm, messages)).lower().strip()

        # Defensive parsing — LLMs occasionally return prose around the word.
        decision: RouteDecision
        if "refuse" in raw:
            decision = "refuse"
        elif "answer_directly" in raw or "direct" in raw:
            decision = "answer_directly"
        else:
            decision = "retrieve"

        log.info("node.route", run_id=state.run_id, decision=decision, raw=raw[:80])
        return {"route": decision}

    return route_node


def _make_rewrite_node():
    llm = build_llm()

    async def rewrite_node(state: AgentState) -> dict:
        if not get_settings().enable_query_rewrite:
            return {"rewritten_query": state.query}

        history = "\n".join(
            f"{m.type}: {m.content}"
            for m in state.messages[-4:]  # last 2 turns
        )
        rewritten = await invoke_with_retry_async(
            llm, build_rewrite_prompt(state.query, history)
        )
        log.info(
            "node.rewrite",
            run_id=state.run_id,
            original=state.query[:60],
            rewritten=rewritten[:60],
        )
        return {"rewritten_query": rewritten}

    return rewrite_node


def _make_retrieve_node(retriever: HybridRetriever):
    async def retrieve_node(state: AgentState) -> dict:
        query = state.rewritten_query or state.query
        # aretrieve pushes the blocking embeddings call + CPU-bound BM25 scan
        # onto a worker thread so the event loop stays responsive.
        chunks = await retriever.aretrieve(query)
        return {"chunks": chunks, "iterations": state.iterations + 1}

    return retrieve_node


def _make_grade_node():
    llm = build_llm()

    async def grade_node(state: AgentState) -> dict:
        if not get_settings().enable_grading or not state.chunks:
            # Skip grading -> assume relevant, let answer node handle empty case.
            grading = GradingResult(
                is_relevant=bool(state.chunks),
                rationale="grading disabled or no chunks",
                confidence=0.5,
            )
            return {"grading": grading}

        raw = await invoke_with_retry_async(
            llm, build_grader_prompt(state.query, state.chunks)
        )
        try:
            # Strip code fences if the LLM added them.
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            payload = json.loads(cleaned)
            grading = GradingResult(
                is_relevant=bool(payload["is_relevant"]),
                rationale=str(payload.get("rationale", "")),
                confidence=float(payload.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            # Fail open: treat as relevant rather than blocking the user.
            log.warning("node.grade.parse_failed", error=str(exc), raw=raw[:120])
            grading = GradingResult(
                is_relevant=True, rationale="grader parse failed; failing open", confidence=0.3
            )

        log.info(
            "node.grade",
            run_id=state.run_id,
            relevant=grading.is_relevant,
            confidence=grading.confidence,
        )
        return {"grading": grading}

    return grade_node


def _make_generate_node():
    llm = build_llm()

    async def generate_node(state: AgentState) -> dict:
        answer = await invoke_with_retry_async(
            llm, build_answer_prompt(state.query, state.chunks)
        )
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return generate_node


async def _direct_answer_node(state: AgentState) -> dict:
    llm = build_llm()
    answer = await invoke_with_retry_async(llm, [("user", state.query)])
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


async def _refuse_node(state: AgentState) -> dict:
    msg = (
        "I can't help with that — it falls outside what this assistant is "
        "designed to do. If you think this was a mistake, rephrase your "
        "question with more context."
    )
    return {"final_answer": msg, "messages": [AIMessage(content=msg)]}


# ---------------------------------------------------------------------------
# Edge logic
# ---------------------------------------------------------------------------


def _route_edge(state: AgentState) -> Literal["retrieve", "answer_directly", "refuse"]:
    return state.route or "retrieve"


def _grade_edge(state: AgentState) -> Literal["generate", "rewrite_query", "generate_no_context"]:
    """After grading, decide whether to retry or commit to an answer."""
    settings = get_settings()
    if state.grading and state.grading.is_relevant:
        return "generate"
    if state.iterations >= settings.max_iterations:
        # Give up retrying — let generate_node tell the user we don't know.
        return "generate_no_context"
    return "rewrite_query"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(retriever: HybridRetriever):
    """Compile and return the LangGraph runnable.

    Accepts the retriever as a dependency so tests can inject a fake.
    """
    builder = StateGraph(AgentState)

    builder.add_node("route", _make_route_node())
    builder.add_node("rewrite_query", _make_rewrite_node())
    builder.add_node("retrieve", _make_retrieve_node(retriever))
    builder.add_node("grade", _make_grade_node())
    builder.add_node("generate", _make_generate_node())
    builder.add_node("direct_answer", _direct_answer_node)
    builder.add_node("refuse", _refuse_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        _route_edge,
        {
            "retrieve": "rewrite_query",
            "answer_directly": "direct_answer",
            "refuse": "refuse",
        },
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        _grade_edge,
        {
            "generate": "generate",
            "generate_no_context": "generate",
            "rewrite_query": "rewrite_query",
        },
    )
    builder.add_edge("generate", END)
    builder.add_edge("direct_answer", END)
    builder.add_edge("refuse", END)

    return builder.compile()
