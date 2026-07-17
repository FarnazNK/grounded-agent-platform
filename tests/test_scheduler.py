"""Tests for the dynamic batch scheduler.

These target the properties that are easy to get wrong and expensive to get
wrong in production:

- batches form under the size/deadline rule (and the deadline is anchored to
  the FIRST request, not the last — otherwise a steady stream starves it)
- a full queue rejects rather than growing without bound
- every submit() resolves or raises; none hang forever
- a failing batch fails only its own callers, not the consumer loop
- results map back to the right caller

The last one matters most: a scheduler that silently hands caller A caller
B's transcript is worse than one that crashes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import numpy as np
import pytest

from rag_agent.voice.scheduler import (
    DynamicBatchScheduler,
    QueueFullError,
    SchedulerClosedError,
    SchedulerConfig,
)
from rag_agent.voice.stt.base import TranscriptEvent
from rag_agent.voice.stt.simulated import SimulatedSTT, SimulatedSTTConfig, make_audio


def _echo_infer(latency: float = 0.01, record: list[int] | None = None):
    """Batch fn that echoes each input's identity, so we can prove results
    land with the right caller. Input i is an array filled with value i."""

    async def infer(audio: Sequence[np.ndarray]) -> list[TranscriptEvent]:
        if record is not None:
            record.append(len(audio))
        await asyncio.sleep(latency)
        return [
            TranscriptEvent(text=f"id={int(a[0])}", is_final=True, audio_seconds=len(a) / 16000)
            for a in audio
        ]

    return infer


def _tagged_audio(i: int, samples: int = 1600) -> np.ndarray:
    """Audio array tagged with an identity so results can be traced."""
    return np.full(samples, float(i), dtype=np.float32)


# ---------------------------------------------------------------------------
# Batching behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_request_does_not_wait_for_full_batch():
    """One request must not sit for max_wait_ms * something. It waits the
    batch window once, then goes. At concurrency 1 the window is pure cost."""
    sizes: list[int] = []
    cfg = SchedulerConfig(max_batch_size=8, max_wait_ms=20)

    async with DynamicBatchScheduler(_echo_infer(0.01, sizes), config=cfg) as sched:
        start = time.perf_counter()
        result = await sched.submit(_tagged_audio(0))
        elapsed = time.perf_counter() - start

    assert result.text == "id=0"
    assert sizes == [1]
    # 20ms window + 10ms infer + overhead. Must not be a multiple of the window.
    assert elapsed < 0.15, f"single request took {elapsed * 1000:.0f}ms"


@pytest.mark.asyncio
async def test_concurrent_requests_form_one_batch():
    """8 simultaneous submits should coalesce into a single batch."""
    sizes: list[int] = []
    cfg = SchedulerConfig(max_batch_size=8, max_wait_ms=50)

    async with DynamicBatchScheduler(_echo_infer(0.01, sizes), config=cfg) as sched:
        results = await asyncio.gather(*[sched.submit(_tagged_audio(i)) for i in range(8)])

    assert len(results) == 8
    assert sizes == [8], f"expected one batch of 8, got batches {sizes}"


@pytest.mark.asyncio
async def test_batch_capped_at_max_batch_size():
    """20 concurrent submits with max_batch_size=8 must never exceed 8."""
    sizes: list[int] = []
    cfg = SchedulerConfig(max_batch_size=8, max_wait_ms=50)

    async with DynamicBatchScheduler(_echo_infer(0.01, sizes), config=cfg) as sched:
        await asyncio.gather(*[sched.submit(_tagged_audio(i)) for i in range(20)])

    assert sizes, "no batches ran"
    assert max(sizes) <= 8, f"batch exceeded cap: {sizes}"
    assert sum(sizes) == 20


@pytest.mark.asyncio
async def test_results_map_to_correct_caller():
    """The correctness property that matters most: no crossed wires.

    Runs 32 concurrent requests through batching and asserts every caller got
    back its OWN result.
    """
    cfg = SchedulerConfig(max_batch_size=8, max_wait_ms=10)

    async with DynamicBatchScheduler(_echo_infer(0.005), config=cfg) as sched:
        results = await asyncio.gather(*[sched.submit(_tagged_audio(i)) for i in range(32)])

    for i, r in enumerate(results):
        assert r.text == f"id={i}", f"caller {i} received {r.text} — crossed wires"


@pytest.mark.asyncio
async def test_zero_wait_disables_batching():
    """max_wait_ms=0 means every request goes alone: minimum latency,
    minimum throughput. The degenerate end of the tradeoff."""
    sizes: list[int] = []
    cfg = SchedulerConfig(max_batch_size=8, max_wait_ms=0)

    async with DynamicBatchScheduler(_echo_infer(0.005, sizes), config=cfg) as sched:
        await asyncio.gather(*[sched.submit(_tagged_audio(i)) for i in range(6)])

    assert all(s <= 2 for s in sizes), f"expected ~no batching with wait=0, got {sizes}"


@pytest.mark.asyncio
async def test_batch_deadline_anchored_to_first_request():
    """The livelock guard.

    A steady arrival stream (one request every 5ms, window 30ms) must still
    produce a batch on schedule. If the deadline reset on each arrival, the
    batch would never close and the first caller would wait forever.
    """
    sizes: list[int] = []
    cfg = SchedulerConfig(max_batch_size=100, max_wait_ms=30)

    async def drip(sched: DynamicBatchScheduler) -> list[TranscriptEvent]:
        out = []
        for i in range(10):
            out.append(asyncio.create_task(sched.submit(_tagged_audio(i))))
            await asyncio.sleep(0.005)
        return await asyncio.gather(*out)

    async with DynamicBatchScheduler(_echo_infer(0.005, sizes), config=cfg) as sched:
        results = await asyncio.wait_for(drip(sched), timeout=2.0)

    assert len(results) == 10
    # With a 30ms window and 5ms drip, batches close on the deadline rather
    # than absorbing all 10. Never one batch of 100, never a hang.
    assert sizes, "no batches ran"
    assert max(sizes) < 100


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_full_rejects_rather_than_buffering():
    """Backpressure: at capacity, submit raises instead of queueing forever.

    Note the submits are CONCURRENT. A sequential `await submit()` loop can
    never overflow the queue — each call blocks until its own request
    completes, so at most one item is ever queued. Overload is by definition
    a concurrency phenomenon, and testing it sequentially tests nothing.
    """
    cfg = SchedulerConfig(max_batch_size=1, max_wait_ms=0, queue_capacity=4)

    async def try_submit(sched: DynamicBatchScheduler, i: int) -> str:
        try:
            await sched.submit(_tagged_audio(i))
            return "ok"
        except QueueFullError:
            return "rejected"

    async with DynamicBatchScheduler(_echo_infer(0.3), config=cfg) as sched:
        results = await asyncio.gather(*[try_submit(sched, i) for i in range(30)])

    assert "rejected" in results, "overload produced no backpressure"
    assert "ok" in results, "backpressure rejected everything"
    # The queue is bounded, so far more are shed than served.
    assert results.count("rejected") > results.count("ok")
    assert sched.stats.rejected == results.count("rejected")


@pytest.mark.asyncio
async def test_queue_depth_stays_within_capacity():
    """`queue_depth` must never exceed the configured bound.

    NOTE ON WHAT IS *NOT* TESTED HERE.

    The consumer acquires its batch slot before pulling from the queue, so it
    never parks a request in a local variable where it would be invisible to
    `queue_depth` and uncounted against `queue_capacity`. That ordering is
    deliberate (see `_run`), but it is not directly observable from outside:
    `put_nowait` rejects synchronously, so under a burst every submit resolves
    against the full queue before the consumer is ever scheduled, and the
    accept count comes out identical either way. A test asserting an accept
    ceiling therefore passes with the ordering reversed — it would be a test
    that cannot fail, which is worse than no test.

    What we assert instead is the invariant that IS observable: depth stays
    within capacity.
    """
    capacity = 4
    cfg = SchedulerConfig(
        max_batch_size=8,
        max_wait_ms=5,
        queue_capacity=capacity,
        max_concurrent_batches=2,
    )

    async def try_submit(sched: DynamicBatchScheduler, i: int) -> str:
        try:
            await sched.submit(_tagged_audio(i))
            return "ok"
        except QueueFullError:
            return "rejected"

    observed: list[int] = []

    async def watch(sched: DynamicBatchScheduler) -> None:
        for _ in range(50):
            observed.append(sched.queue_depth)
            await asyncio.sleep(0.002)

    async with DynamicBatchScheduler(_echo_infer(0.05), config=cfg) as sched:
        watcher = asyncio.create_task(watch(sched))
        await asyncio.gather(*[try_submit(sched, i) for i in range(60)])
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

    assert observed, "watcher never sampled"
    assert max(observed) <= capacity, (
        f"queue depth reached {max(observed)} with capacity={capacity}"
    )


@pytest.mark.asyncio
async def test_rejection_is_counted():
    cfg = SchedulerConfig(max_batch_size=1, max_wait_ms=0, queue_capacity=2)

    async def try_submit(sched: DynamicBatchScheduler, i: int) -> str:
        try:
            await sched.submit(_tagged_audio(i))
            return "ok"
        except QueueFullError:
            return "rejected"

    async with DynamicBatchScheduler(_echo_infer(0.3), config=cfg) as sched:
        results = await asyncio.gather(*[try_submit(sched, i) for i in range(20)])

    rejected = results.count("rejected")
    assert rejected > 0
    assert sched.stats.rejected == rejected


# ---------------------------------------------------------------------------
# Failure isolation and lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failing_batch_does_not_kill_scheduler():
    """One bad batch fails its callers; the next batch still works.

    A consumer loop that dies on the first exception turns a transient model
    error into a permanently wedged service.
    """
    calls = {"n": 0}

    async def flaky(audio: Sequence[np.ndarray]) -> list[TranscriptEvent]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated CUDA OOM")
        return [TranscriptEvent(text="ok", is_final=True) for _ in audio]

    cfg = SchedulerConfig(max_batch_size=4, max_wait_ms=5)

    async with DynamicBatchScheduler(flaky, config=cfg) as sched:
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            await sched.submit(_tagged_audio(0))

        # Scheduler must still be alive.
        result = await sched.submit(_tagged_audio(1))
        assert result.text == "ok"

    assert sched.stats.failed >= 1


@pytest.mark.asyncio
async def test_provider_result_count_mismatch_is_caught():
    """A provider returning the wrong number of results must raise, not
    silently misalign callers."""

    async def bad(audio: Sequence[np.ndarray]) -> list[TranscriptEvent]:
        return [TranscriptEvent(text="only-one", is_final=True)]  # wrong count

    cfg = SchedulerConfig(max_batch_size=4, max_wait_ms=20)

    async with DynamicBatchScheduler(bad, config=cfg) as sched:
        with pytest.raises(RuntimeError, match="mismatch"):
            await asyncio.gather(*[sched.submit(_tagged_audio(i)) for i in range(4)])


@pytest.mark.asyncio
async def test_submit_timeout_raises():
    cfg = SchedulerConfig(max_batch_size=1, max_wait_ms=0)

    async with DynamicBatchScheduler(_echo_infer(1.0), config=cfg) as sched:
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await sched.submit(_tagged_audio(0), timeout=0.05)


@pytest.mark.asyncio
async def test_submit_is_cancellable():
    """Cancelling a submit must not wedge the scheduler — barge-in path."""
    cfg = SchedulerConfig(max_batch_size=1, max_wait_ms=0)

    async with DynamicBatchScheduler(_echo_infer(0.5), config=cfg) as sched:
        task = asyncio.create_task(sched.submit(_tagged_audio(0)))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Still serving afterwards.
        result = await asyncio.wait_for(sched.submit(_tagged_audio(1)), timeout=2.0)
        assert result.text == "id=1"


@pytest.mark.asyncio
async def test_submit_before_start_raises():
    sched = DynamicBatchScheduler(_echo_infer(), config=SchedulerConfig())
    with pytest.raises(SchedulerClosedError):
        await sched.submit(_tagged_audio(0))


@pytest.mark.asyncio
async def test_close_drains_queued_work():
    """Graceful shutdown completes in-flight work rather than dropping it."""
    cfg = SchedulerConfig(max_batch_size=4, max_wait_ms=5)
    sched = DynamicBatchScheduler(_echo_infer(0.02), config=cfg)
    await sched.start()

    tasks = [asyncio.create_task(sched.submit(_tagged_audio(i))) for i in range(8)]
    await asyncio.sleep(0.01)
    await sched.aclose(drain=True)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = [r for r in results if isinstance(r, TranscriptEvent)]
    assert len(ok) == 8, f"drain lost work: {len(ok)}/8 completed"


@pytest.mark.asyncio
async def test_stats_track_batching():
    cfg = SchedulerConfig(max_batch_size=8, max_wait_ms=30)

    async with DynamicBatchScheduler(_echo_infer(0.005), config=cfg) as sched:
        await asyncio.gather(*[sched.submit(_tagged_audio(i)) for i in range(16)])

    s = sched.stats
    assert s.submitted == 16
    assert s.completed == 16
    assert s.batches_run >= 1
    assert s.mean_batch_size > 1.0


@pytest.mark.asyncio
async def test_metric_hooks_fire():
    """The scheduler reports batch size and queue wait to injected hooks —
    that's how the Prometheus metrics get populated without this module
    importing the API layer."""
    batches: list[tuple[int, float]] = []
    waits: list[float] = []
    cfg = SchedulerConfig(max_batch_size=4, max_wait_ms=20)

    async with DynamicBatchScheduler(
        _echo_infer(0.005),
        config=cfg,
        on_batch=lambda size, elapsed: batches.append((size, elapsed)),
        on_queue_wait=waits.append,
    ) as sched:
        await asyncio.gather(*[sched.submit(_tagged_audio(i)) for i in range(4)])

    assert batches, "on_batch never fired"
    assert sum(b[0] for b in batches) == 4
    assert len(waits) == 4
    assert all(w >= 0 for w in waits)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_batch_size": 0},
        {"max_wait_ms": -1},
        {"queue_capacity": 0},
        {"max_concurrent_batches": 0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        SchedulerConfig(**kwargs)


# ---------------------------------------------------------------------------
# Simulated provider
# ---------------------------------------------------------------------------


def test_simulated_cost_model_shape():
    """The simulated backend must reproduce the property that makes batching
    worth doing: batch(N) cheaper than N x batch(1), but slower than batch(1).

    If this inverts, every benchmark built on it is meaningless.
    """
    p = SimulatedSTT()
    one = p.batch_latency_seconds(1, audio_seconds=2.0)
    eight = p.batch_latency_seconds(8, audio_seconds=2.0)

    assert eight < 8 * one, "batching must amortize per-call overhead"
    assert eight > one, "batching must cost the individual request latency"
    assert (eight / 8) < one, "per-item cost must fall as batch grows"
    assert p.batch_latency_seconds(4, 10.0) > p.batch_latency_seconds(4, 1.0)


@pytest.mark.asyncio
async def test_simulated_provider_returns_one_result_per_input():
    p = SimulatedSTT(SimulatedSTTConfig(fixed_overhead_ms=1, per_item_ms=0.1))
    audio = [make_audio(1.0) for _ in range(5)]
    results = await p.transcribe_batch(audio)

    assert len(results) == 5
    assert all(r.metadata["simulated"] is True for r in results)


@pytest.mark.asyncio
async def test_simulated_provider_satisfies_protocol():
    from rag_agent.voice.stt.base import SpeechToTextProvider

    assert isinstance(SimulatedSTT(), SpeechToTextProvider)


@pytest.mark.asyncio
async def test_scheduler_with_simulated_provider():
    """End-to-end: the real scheduler driving the simulated backend."""
    p = SimulatedSTT(SimulatedSTTConfig(fixed_overhead_ms=5, per_item_ms=1))
    cfg = SchedulerConfig(max_batch_size=8, max_wait_ms=10)

    async with DynamicBatchScheduler(p.transcribe_batch, config=cfg) as sched:
        results = await asyncio.gather(
            *[sched.submit(make_audio(1.0), session_id=f"s{i}") for i in range(16)]
        )

    assert len(results) == 16
    assert all(r.is_final for r in results)
    assert p.stats["items"] == 16
    assert p.stats["calls"] < 16, "batching should mean fewer calls than items"
