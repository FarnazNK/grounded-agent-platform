"""Tests for the LangGraph wiring.

These stub out the LLM with a deterministic fake so the whole graph can run
in CI without any API keys. We're testing topology and state-threading, not
the LLM itself.

The graph is fully async, so the fake replaces `invoke_with_retry_async` and
the graph is driven with `ainvoke`. Each test asserts the same topology
invariants as before the async conversion.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

from langchain_core.documents import Document

from rag_agent.graph import build_graph
from rag_agent.retrieval import HybridRetriever
from rag_agent.schemas import AgentState, RetrievedChunk


class FakeStore:
    """In-memory vector store that returns canned results."""

    def __init__(self, canned: list[RetrievedChunk] | None = None) -> None:
        self._canned = canned or []

    def add_documents(self, docs):
        return []

    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._canned[:k]

    def count(self) -> int:
        return len(self._canned)


def _fake_invoke_factory(responses: dict[str, str]):
    """Build a fake `invoke_with_retry_async` that returns different strings
    based on what system prompt it sees. Lets one fake serve every node.

    Async because the nodes now await it — patching in a sync function would
    hand the node a str where it awaits a coroutine.
    """

    async def fake_invoke(llm: Any, messages: list[tuple[str, str]]) -> str:
        system = next((m[1] for m in messages if m[0] == "system"), "")
        for key, reply in responses.items():
            if key in system:
                return reply
        # Fallback for the direct-answer path (no system prompt).
        return responses.get("__default__", "ok")

    return fake_invoke


def _corpus() -> list[Document]:
    return [
        Document(
            page_content="Employees accrue 15 PTO days per year in their first 2 years.",
            metadata={"source": "pto.md", "chunk_id": "pto-0"},
        ),
        Document(
            page_content="Performance reviews run twice per year.",
            metadata={"source": "reviews.md", "chunk_id": "rev-0"},
        ),
    ]


def test_retrieval_path_reaches_generate():
    """Happy path: router says retrieve, grader says relevant, answer produced."""
    canned = [
        RetrievedChunk(
            chunk_id="pto-0",
            content="Employees accrue 15 PTO days per year in their first 2 years.",
            source="pto.md",
            score=0.9,
        )
    ]
    retriever = HybridRetriever(store=FakeStore(canned), corpus=_corpus())

    fake_invoke = _fake_invoke_factory(
        {
            "router": "retrieve",
            "rewrite": "how many pto days",
            "grade": '{"is_relevant": true, "confidence": 0.9, "rationale": "direct match"}',
            "helpful assistant": "New hires get 15 PTO days per year. [source: pto.md]",
        }
    )

    with patch("rag_agent.graph.invoke_with_retry_async", side_effect=fake_invoke):
        graph = build_graph(retriever)
        result = asyncio.run(graph.ainvoke(AgentState(query="How many PTO days?")))

    state = AgentState.model_validate(result)
    assert state.route == "retrieve"
    assert state.final_answer is not None
    assert "15 PTO days" in state.final_answer
    assert len(state.chunks) >= 1
    assert state.iterations >= 1


def test_refuse_path_short_circuits():
    """Router=refuse should skip retrieval and grading entirely."""
    retriever = HybridRetriever(store=FakeStore(), corpus=_corpus())

    fake_invoke = _fake_invoke_factory({"router": "refuse"})

    with patch("rag_agent.graph.invoke_with_retry_async", side_effect=fake_invoke):
        graph = build_graph(retriever)
        result = asyncio.run(graph.ainvoke(AgentState(query="Do something harmful")))

    state = AgentState.model_validate(result)
    assert state.route == "refuse"
    assert state.iterations == 0  # never retrieved
    assert state.chunks == []
    assert state.final_answer  # refusal message populated


def test_irrelevant_retrieval_triggers_retry_then_gives_up():
    """When grader says irrelevant, graph loops up to max_iterations."""
    canned = [
        RetrievedChunk(
            chunk_id="rev-0",
            content="Performance reviews run twice per year.",
            source="reviews.md",
            score=0.5,
        )
    ]
    retriever = HybridRetriever(store=FakeStore(canned), corpus=_corpus())

    fake_invoke = _fake_invoke_factory(
        {
            "router": "retrieve",
            "rewrite": "rewritten query",
            # Always say irrelevant — forces the loop to exhaust itself.
            "grade": '{"is_relevant": false, "confidence": 0.9, "rationale": "off-topic"}',
            "helpful assistant": "I don't have that information.",
        }
    )

    with patch("rag_agent.graph.invoke_with_retry_async", side_effect=fake_invoke):
        graph = build_graph(retriever)
        result = asyncio.run(graph.ainvoke(AgentState(query="What's our parental leave policy?")))

    state = AgentState.model_validate(result)
    # Iterations should hit the cap (default 3) then generate anyway.
    assert state.iterations >= 3
    assert state.final_answer is not None
    assert state.grading is not None
    assert state.grading.is_relevant is False
