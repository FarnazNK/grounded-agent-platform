"""Wiring between the scheduler and Prometheus.

Lives here rather than in `scheduler.py` so the scheduler has no dependency
on the API layer — it takes plain callables and doesn't know Prometheus
exists. That keeps it unit-testable without a registry, and means a second
consumer (a CLI, a different service) can instrument it differently.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from rag_agent.api.metrics import (
    AUDIO_SECONDS_PROCESSED,
    INFERENCE_BATCH_SIZE,
    INFERENCE_QUEUE_WAIT,
    REQUESTS_IN_FLIGHT,
    STAGE_LATENCY,
    TRANSCRIPTION_RTF,
    VOICE_ERRORS,
)
from rag_agent.voice.scheduler import DynamicBatchScheduler, SchedulerConfig
from rag_agent.voice.stt.base import SpeechToTextProvider


def record_batch(size: int, elapsed_seconds: float) -> None:
    """Scheduler `on_batch` hook."""
    INFERENCE_BATCH_SIZE.observe(size)
    STAGE_LATENCY.labels(stage="stt").observe(elapsed_seconds)


def record_queue_wait(wait_seconds: float) -> None:
    """Scheduler `on_queue_wait` hook."""
    INFERENCE_QUEUE_WAIT.observe(wait_seconds)


def build_instrumented_scheduler(
    provider: SpeechToTextProvider,
    config: SchedulerConfig | None = None,
) -> DynamicBatchScheduler:
    """A scheduler wired to the Prometheus registry."""
    return DynamicBatchScheduler(
        provider.transcribe_batch,
        config=config,
        on_batch=record_batch,
        on_queue_wait=record_queue_wait,
    )


@contextmanager
def stage_timer(stage: str) -> Iterator[None]:
    """Time one pipeline stage and count its failures.

    Usage:
        with stage_timer("agent"):
            state = await agent.arun(text)
    """
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        VOICE_ERRORS.labels(stage=stage, type=type(exc).__name__).inc()
        raise
    finally:
        STAGE_LATENCY.labels(stage=stage).observe(time.perf_counter() - start)


@contextmanager
def track_in_flight() -> Iterator[None]:
    """Gauge for concurrent inference requests."""
    REQUESTS_IN_FLIGHT.inc()
    try:
        yield
    finally:
        REQUESTS_IN_FLIGHT.dec()


def record_transcription(audio_seconds: float, compute_seconds: float) -> None:
    """Record throughput and real-time factor for a completed transcription.

    RTF is the metric that answers "can one worker keep up with one speaker?"
    Anything >= 1.0 means it cannot, and the backlog grows without bound.
    """
    if audio_seconds <= 0:
        return
    AUDIO_SECONDS_PROCESSED.inc(audio_seconds)
    TRANSCRIPTION_RTF.observe(compute_seconds / audio_seconds)
