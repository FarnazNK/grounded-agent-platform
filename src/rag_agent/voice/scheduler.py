"""Inference scheduler: bounded queue + dynamic batching.

This is the heart of the serving layer. Callers submit one audio segment and
await one transcript; the scheduler decides how to group those segments into
batches before they reach the model.

Why batch at all
----------------
GPU inference is dominated by fixed per-call overhead — kernel launches, and
host-to-device transfer. Running 8 segments as one batch costs far less than
8x a single segment, so batching buys throughput. It costs latency: the first
request in a batch waits for the batch to fill. That tradeoff is the whole
design, and `max_batch_size` / `max_wait_ms` are the two knobs that set it.

The batching rule
-----------------
A batch closes when EITHER:
- it reaches `max_batch_size`, or
- `max_wait_ms` elapses since the batch's FIRST request arrived.

The deadline is anchored to the first request, not the last. Anchoring to the
last would let a steady arrival stream postpone the batch forever and starve
the request that has already been waiting longest — the classic livelock in
naive batchers.

Backpressure
------------
The queue is bounded. When it is full, `submit` raises `QueueFullError`
rather than growing memory without limit. That is a deliberate choice: an
overloaded voice service should shed load and tell the client, because an
unbounded queue converts an overload into an OOM, and every queued request
whose caller has already hung up is pure waste. Rejecting fast is a feature.

Concurrency model
-----------------
A single consumer task owns the queue and runs batches. `max_concurrent_batches`
allows N batches in flight at once so the next batch can be assembled while
the previous one is on the device. Beyond that, requests queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rag_agent.observability import get_logger
from rag_agent.voice.stt.base import DEFAULT_SAMPLE_RATE, TranscriptEvent

log = get_logger(__name__)

# Callable the scheduler drives. Matches SpeechToTextProvider.transcribe_batch.
BatchInferFn = Callable[[Sequence[np.ndarray]], Awaitable[list[TranscriptEvent]]]


class QueueFullError(RuntimeError):
    """Raised when the queue is at capacity — the backpressure signal.

    Callers should map this to a 503 (HTTP) or a session error event (WS),
    not retry blindly: the queue being full means the service is already
    behind, and retrying makes it worse.
    """


class SchedulerClosedError(RuntimeError):
    """Raised when submitting to a scheduler that is shutting down."""


@dataclass
class SchedulerConfig:
    """Batching knobs. These are the parameters the benchmark sweeps."""

    max_batch_size: int = 8
    max_wait_ms: float = 15.0
    queue_capacity: int = 256
    max_concurrent_batches: int = 1
    sample_rate: int = DEFAULT_SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")
        if self.max_concurrent_batches < 1:
            raise ValueError("max_concurrent_batches must be >= 1")


@dataclass
class InferenceRequest:
    """One unit of work: audio in, a future the caller awaits."""

    audio: np.ndarray
    future: asyncio.Future[TranscriptEvent]
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    enqueued_at: float = field(default_factory=time.perf_counter)
    deadline: float | None = None

    @property
    def queue_wait_seconds(self) -> float:
        return time.perf_counter() - self.enqueued_at

    def is_expired(self, now: float | None = None) -> bool:
        """True when this request's caller has already given up.

        Checked before inference: computing a transcript nobody will read
        burns the exact GPU time the live requests are queued for.
        """
        if self.deadline is None:
            return False
        return (now or time.perf_counter()) >= self.deadline


@dataclass
class SchedulerStats:
    """Cumulative counters. Cheap to read, exported to Prometheus."""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    expired: int = 0
    batches_run: int = 0
    total_batched_items: int = 0

    @property
    def mean_batch_size(self) -> float:
        if self.batches_run == 0:
            return 0.0
        return self.total_batched_items / self.batches_run


class DynamicBatchScheduler:
    """Groups concurrent inference requests into batches.

    Usage:
        sched = DynamicBatchScheduler(provider.transcribe_batch, config=cfg)
        await sched.start()
        event = await sched.submit(audio)
        await sched.aclose()

    Or as a context manager:
        async with DynamicBatchScheduler(fn, config=cfg) as sched:
            event = await sched.submit(audio)
    """

    def __init__(
        self,
        infer_batch: BatchInferFn,
        *,
        config: SchedulerConfig | None = None,
        on_batch: Callable[[int, float], None] | None = None,
        on_queue_wait: Callable[[float], None] | None = None,
    ) -> None:
        self._infer_batch = infer_batch
        self._config = config or SchedulerConfig()
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(
            maxsize=self._config.queue_capacity
        )
        self._consumer: asyncio.Task[None] | None = None
        self._in_flight: set[asyncio.Task[None]] = set()
        self._batch_slots = asyncio.Semaphore(self._config.max_concurrent_batches)
        self._closing = False
        self._stats = SchedulerStats()

        # Metric hooks. Injected rather than imported so the scheduler stays
        # testable without a Prometheus registry, and so this module doesn't
        # depend on the API layer.
        self._on_batch = on_batch
        self._on_queue_wait = on_queue_wait

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the consumer loop. Idempotent."""
        if self._consumer is None or self._consumer.done():
            self._closing = False
            self._consumer = asyncio.create_task(self._run(), name="batch-scheduler")
            log.info(
                "scheduler.started",
                max_batch_size=self._config.max_batch_size,
                max_wait_ms=self._config.max_wait_ms,
                queue_capacity=self._config.queue_capacity,
                max_concurrent_batches=self._config.max_concurrent_batches,
            )

    async def aclose(self, *, drain: bool = True) -> None:
        """Stop accepting work and shut down.

        With `drain=True` (default) queued requests are allowed to finish —
        this is what a graceful rollout needs, so in-flight sessions aren't
        severed mid-utterance when a pod is being replaced.
        """
        self._closing = True
        if drain:
            await self._queue.join()

        if self._consumer is not None:
            self._consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer
            self._consumer = None

        if self._in_flight:
            await asyncio.gather(*self._in_flight, return_exceptions=True)

        # Anything still queued after a non-draining close gets an explicit
        # error rather than a future that never resolves.
        while not self._queue.empty():
            req = self._queue.get_nowait()
            if not req.future.done():
                req.future.set_exception(SchedulerClosedError("scheduler closed"))
            self._queue.task_done()

        log.info("scheduler.closed", **self.stats_dict())

    async def __aenter__(self) -> DynamicBatchScheduler:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose(drain=False)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit(
        self,
        audio: np.ndarray,
        *,
        session_id: str = "",
        timeout: float | None = None,
    ) -> TranscriptEvent:
        """Submit one segment and await its transcript.

        Raises:
            QueueFullError: queue at capacity — shed load, don't buffer.
            SchedulerClosedError: scheduler is shutting down.
            asyncio.TimeoutError: `timeout` elapsed before completion.
        """
        if self._closing:
            raise SchedulerClosedError("scheduler is shutting down")
        if self._consumer is None:
            raise SchedulerClosedError("scheduler not started — call start()")

        loop = asyncio.get_running_loop()
        req = InferenceRequest(
            audio=audio,
            future=loop.create_future(),
            session_id=session_id,
            deadline=(time.perf_counter() + timeout) if timeout else None,
        )

        try:
            # put_nowait, not await put: awaiting a full queue is silent
            # unbounded latency, which is the failure mode backpressure
            # exists to prevent. Fail fast and let the caller decide.
            self._queue.put_nowait(req)
        except asyncio.QueueFull as exc:
            self._stats.rejected += 1
            log.warning("scheduler.rejected", depth=self._queue.qsize())
            raise QueueFullError(
                f"inference queue full (capacity={self._config.queue_capacity})"
            ) from exc

        self._stats.submitted += 1

        if timeout is None:
            return await req.future
        try:
            return await asyncio.wait_for(req.future, timeout=timeout)
        except TimeoutError:
            # The consumer will notice the expired deadline and skip it.
            raise

    # ------------------------------------------------------------------
    # Consumer loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Collect batches and dispatch them until cancelled.

        The slot is acquired BEFORE collecting a batch, not after. If we
        collected first and then blocked on the semaphore, the consumer would
        sit holding requests that have already left the queue — they'd be
        invisible to `queue_depth` and wouldn't count against
        `queue_capacity`. Effective capacity would silently exceed the
        configured bound, and the backpressure signal would fire late and
        under-report the real backlog. Acquiring first keeps every waiting
        request inside the queue, where it is both counted and bounded.
        """
        try:
            while True:
                await self._batch_slots.acquire()
                try:
                    batch = await self._collect_batch()
                except asyncio.CancelledError:
                    self._batch_slots.release()
                    raise
                if not batch:
                    self._batch_slots.release()
                    continue

                task = asyncio.create_task(self._run_batch(batch))
                self._in_flight.add(task)
                task.add_done_callback(self._in_flight.discard)
                task.add_done_callback(lambda _: self._batch_slots.release())
        except asyncio.CancelledError:
            raise

    async def _collect_batch(self) -> list[InferenceRequest]:
        """Assemble one batch under the size/deadline rule.

        Blocks until at least one request arrives, then keeps pulling until
        the batch is full or the first request's wait budget is spent.
        """
        first = await self._queue.get()
        batch = [first]

        # Deadline anchored to the FIRST request — see module docstring.
        deadline = time.perf_counter() + (self._config.max_wait_ms / 1000.0)

        while len(batch) < self._config.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except TimeoutError:
                break
            batch.append(req)

        return batch

    async def _run_batch(self, batch: list[InferenceRequest]) -> None:
        """Run one batch and resolve every caller's future.

        Every exit path must resolve or reject each future, or `submit`
        hangs forever. That's why this is one big try/except.
        """
        now = time.perf_counter()

        # Drop requests whose callers already timed out before spending
        # device time on them.
        live: list[InferenceRequest] = []
        for req in batch:
            if req.is_expired(now):
                self._stats.expired += 1
                if not req.future.done():
                    req.future.set_exception(TimeoutError("request expired before inference"))
                self._queue.task_done()
            else:
                live.append(req)

        if not live:
            return

        for req in live:
            wait = req.queue_wait_seconds
            if self._on_queue_wait is not None:
                self._on_queue_wait(wait)

        started = time.perf_counter()
        try:
            results = await self._infer_batch([r.audio for r in live])

            # A provider that returns the wrong count would silently hand
            # callers each other's transcripts. Fail loudly instead.
            if len(results) != len(live):
                raise RuntimeError(
                    f"provider returned {len(results)} results for "
                    f"{len(live)} inputs — batch/result mismatch"
                )

            elapsed = time.perf_counter() - started
            self._stats.batches_run += 1
            self._stats.total_batched_items += len(live)
            if self._on_batch is not None:
                self._on_batch(len(live), elapsed)

            for req, result in zip(live, results, strict=True):
                self._stats.completed += 1
                if not req.future.done():
                    req.future.set_result(result)

            log.debug(
                "scheduler.batch_done",
                size=len(live),
                elapsed_ms=elapsed * 1000,
            )
        except asyncio.CancelledError:
            for req in live:
                if not req.future.done():
                    req.future.cancel()
            raise
        except Exception as exc:
            # One bad batch must not kill the consumer loop — fail just these
            # callers and keep serving.
            log.exception("scheduler.batch_failed", error=str(exc), size=len(live))
            self._stats.failed += len(live)
            for req in live:
                if not req.future.done():
                    req.future.set_exception(exc)
        finally:
            for _ in live:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> SchedulerStats:
        return self._stats

    def stats_dict(self) -> dict[str, float | int]:
        s = self._stats
        return {
            "submitted": s.submitted,
            "completed": s.completed,
            "failed": s.failed,
            "rejected": s.rejected,
            "expired": s.expired,
            "batches_run": s.batches_run,
            "mean_batch_size": round(s.mean_batch_size, 2),
            "queue_depth": self.queue_depth,
        }
