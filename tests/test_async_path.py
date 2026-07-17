"""Tests for the async execution path.

These exist to lock in one property: no coroutine on the serving path may
block the event loop. That property is invisible in a single-request test —
a blocking call and an awaiting call look identical at concurrency 1 — so
these tests assert it under concurrency, where it actually shows up.

If someone reintroduces `agent.run()` in an async route, or swaps an
`ainvoke` back to `invoke`, `test_concurrent_arun_overlaps` fails.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from rag_agent.agent import Agent
from rag_agent.retrieval import HybridRetriever
from rag_agent.schemas import AgentState, RetrievedChunk

# Long enough that serialization is unambiguous, short enough for fast CI.
FAKE_LLM_LATENCY = 0.05


class FakeStore:
    """In-memory vector store returning canned results."""

    def __init__(self, canned: list[RetrievedChunk] | None = None) -> None:
        self._canned = canned or []

    def add_documents(self, docs: list[Document]) -> list[str]:
        return []

    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._canned[:k]

    def count(self) -> int:
        return len(self._canned)


def _corpus() -> list[Document]:
    return [
        Document(
            page_content="Employees accrue 15 PTO days per year.",
            metadata={"source": "pto.md", "chunk_id": "pto-0"},
        )
    ]


def _canned_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="pto-0",
            content="Employees accrue 15 PTO days per year.",
            source="pto.md",
            score=0.9,
        )
    ]


def _slow_fake_llm(responses: dict[str, str]):
    """Async fake that sleeps to simulate provider latency.

    `asyncio.sleep` is the point: it yields control. If a node awaits this
    properly, N concurrent runs overlap. If a node blocks, they serialize.
    """

    async def fake_invoke(llm: Any, messages: list[tuple[str, str]]) -> str:
        await asyncio.sleep(FAKE_LLM_LATENCY)
        system = next((m[1] for m in messages if m[0] == "system"), "")
        for key, reply in responses.items():
            if key in system:
                return reply
        return responses.get("__default__", "ok")

    return fake_invoke


_RESPONSES = {
    "router": "retrieve",
    "rewrite": "how many pto days",
    "grade": '{"is_relevant": true, "confidence": 0.9, "rationale": "match"}',
    "helpful assistant": "New hires get 15 PTO days per year. [source: pto.md]",
}


def _build_agent() -> Agent:
    retriever = HybridRetriever(store=FakeStore(_canned_chunks()), corpus=_corpus())
    return Agent(retriever)


@pytest.mark.asyncio
async def test_arun_returns_final_state():
    """arun produces the same shape as run — async is a transport change,
    not a behavior change."""
    with patch("rag_agent.graph.invoke_with_retry_async", side_effect=_slow_fake_llm(_RESPONSES)):
        agent = _build_agent()
        state = await agent.arun("How many PTO days?")

    assert isinstance(state, AgentState)
    assert state.route == "retrieve"
    assert state.final_answer is not None
    assert "15 PTO days" in state.final_answer
    assert state.iterations >= 1


@pytest.mark.asyncio
async def test_aask_returns_answer_string():
    with patch("rag_agent.graph.invoke_with_retry_async", side_effect=_slow_fake_llm(_RESPONSES)):
        agent = _build_agent()
        answer = await agent.aask("How many PTO days?")

    assert isinstance(answer, str)
    assert "15 PTO days" in answer


@pytest.mark.asyncio
async def test_concurrent_arun_overlaps():
    """The load-bearing test.

    8 concurrent runs, each making 4 sequential LLM calls of 50 ms. If the
    graph awaits properly they overlap and total ~4*50ms plus overhead. If
    any node blocks the loop they serialize to ~8*4*50ms = 1.6 s.

    The threshold is set well above the ideal (200 ms) and well below the
    serialized case, so it fails loudly on a real regression without
    flaking on a slow CI box.
    """
    concurrency = 8

    with patch("rag_agent.graph.invoke_with_retry_async", side_effect=_slow_fake_llm(_RESPONSES)):
        agent = _build_agent()

        start = time.perf_counter()
        results = await asyncio.gather(
            *[agent.arun(f"How many PTO days? #{i}") for i in range(concurrency)]
        )
        elapsed = time.perf_counter() - start

    assert len(results) == concurrency
    assert all(r.final_answer for r in results)

    serialized_estimate = concurrency * 4 * FAKE_LLM_LATENCY  # ~1.6s
    assert elapsed < serialized_estimate / 2, (
        f"{concurrency} concurrent runs took {elapsed:.2f}s; "
        f"serialized would be ~{serialized_estimate:.2f}s. "
        "Something on the graph path is blocking the event loop."
    )


@pytest.mark.asyncio
async def test_arun_timeout_cancels():
    """A run that exceeds its deadline raises rather than pinning a worker."""

    async def very_slow(llm: Any, messages: list[tuple[str, str]]) -> str:
        await asyncio.sleep(5.0)
        return "too late"

    with patch("rag_agent.graph.invoke_with_retry_async", side_effect=very_slow):
        agent = _build_agent()
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await agent.arun("How many PTO days?", timeout=0.1)


@pytest.mark.asyncio
async def test_arun_is_cancellable():
    """Cancelling the task holding arun stops the in-flight run.

    This is the mechanism barge-in relies on: when the user starts talking
    over the agent, we cancel the task rather than letting it run to
    completion against a client that stopped listening.
    """
    started = asyncio.Event()

    async def slow(llm: Any, messages: list[tuple[str, str]]) -> str:
        started.set()
        await asyncio.sleep(5.0)
        return "never"

    with patch("rag_agent.graph.invoke_with_retry_async", side_effect=slow):
        agent = _build_agent()
        task = asyncio.create_task(agent.arun("How many PTO days?"))
        await started.wait()  # ensure we're mid-run, not pre-start
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert task.cancelled()


@pytest.mark.asyncio
async def test_aretrieve_offloads_and_matches_sync():
    """aretrieve returns the same chunks as retrieve — the thread offload is
    a scheduling detail, not a semantic one."""
    retriever = HybridRetriever(store=FakeStore(_canned_chunks()), corpus=_corpus())

    sync_result = retriever.retrieve("pto days")
    async_result = await retriever.aretrieve("pto days")

    assert [c.chunk_id for c in sync_result] == [c.chunk_id for c in async_result]


class SlowStore:
    """Vector store whose search burns wall-clock on the calling thread.

    Stands in for the real blocking cost of `similarity_search` (a synchronous
    embeddings HTTP call plus a Chroma query), which in tests would otherwise
    be instant. `time.sleep` here is deliberate: it is the thing an event loop
    cannot survive, so it makes the offload property testable without
    depending on how fast BM25 happens to be on the CI runner.
    """

    BLOCK_SECONDS = 0.05

    def add_documents(self, docs: list[Document]) -> list[str]:
        return []

    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        time.sleep(self.BLOCK_SECONDS)
        return []

    def count(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_aretrieve_does_not_block_loop():
    """While aretrieve runs, the loop must still make progress.

    Retrieval does two things that block the calling thread: a synchronous
    embeddings/Chroma call, and a CPU-bound BM25 scan. `aretrieve` must keep
    both off the loop thread.

    We measure the worst gap between heartbeat ticks. A tick count is a bad
    signal (it also drops when retrieval is merely fast); a single long gap is
    unambiguous evidence the loop was stalled.
    """
    retriever = HybridRetriever(store=SlowStore(), corpus=_corpus())

    max_gap = 0.0
    stop = False

    async def heartbeat():
        nonlocal max_gap
        last = time.perf_counter()
        while not stop:
            await asyncio.sleep(0)  # yield; reschedule immediately
            now = time.perf_counter()
            max_gap = max(max_gap, now - last)
            last = now

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat establish a baseline
    await asyncio.gather(*[retriever.aretrieve("pto days") for _ in range(8)])
    stop = True
    await hb

    # Offloaded: the loop keeps getting scheduled, gaps stay near zero.
    # On the loop thread: 8 x 50 ms of sleep serializes into one long stall.
    # 25 ms sits well above scheduler jitter and well below one BLOCK_SECONDS.
    assert max_gap < 0.025, (
        f"event loop stalled {max_gap * 1000:.1f} ms during retrieval — "
        "blocking retrieval work is running on the loop thread"
    )


@pytest.mark.asyncio
async def test_blocking_retrieve_would_stall_loop():
    """Control for the test above.

    Asserts the probe can actually detect a stall. Without this, a threshold
    that never fires would look identical to a passing test. Here we call the
    *sync* retrieve on the loop on purpose and require the probe to catch it.
    """
    retriever = HybridRetriever(store=SlowStore(), corpus=_corpus())

    max_gap = 0.0
    stop = False

    async def heartbeat():
        nonlocal max_gap
        last = time.perf_counter()
        while not stop:
            await asyncio.sleep(0)
            now = time.perf_counter()
            max_gap = max(max_gap, now - last)
            last = now

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)

    # The regression, performed deliberately: sync retrieve, no offload.
    for _ in range(2):
        retriever.retrieve("pto days")

    stop = True
    await hb

    assert max_gap >= 0.025, (
        "probe failed to detect a known-blocking call — "
        "test_aretrieve_does_not_block_loop is toothless"
    )
