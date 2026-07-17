"""Prometheus metrics for the API and the voice serving path.

Two families here:

1. RAG/HTTP metrics — request count by route and status, end-to-end latency,
   guardrail blocks, loop iterations.

2. Voice serving metrics — stage latency, queue depth, batch size, RTF,
   cancellations. These exist because request-level latency cannot answer the
   only question that matters during a p99 regression: which stage?

Everything registers against a custom registry rather than the global one so
multiple test instances don't collide.

Metric naming follows Prometheus convention: base unit (seconds, not ms),
`_total` on counters, and histograms bucketed for the range we actually care
about rather than the library default.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Shared registry — reset in tests by re-importing the module.
REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "rag_agent_requests_total",
    "Total HTTP requests handled by the API.",
    labelnames=("endpoint", "status"),
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "rag_agent_request_latency_seconds",
    "End-to-end request latency in seconds.",
    labelnames=("endpoint",),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

GUARDRAIL_BLOCKS = Counter(
    "rag_agent_guardrail_blocks_total",
    "Number of times a guardrail blocked a request.",
    labelnames=("guardrail",),
    registry=REGISTRY,
)

GUARDRAIL_SANITIZATIONS = Counter(
    "rag_agent_guardrail_sanitizations_total",
    "Number of times a guardrail sanitized a request.",
    labelnames=("guardrail",),
    registry=REGISTRY,
)

ITERATIONS_OBSERVED = Histogram(
    "rag_agent_loop_iterations",
    "Number of retrieve-grade loops per request.",
    buckets=(0, 1, 2, 3, 4, 5, 6),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Voice serving metrics
# ---------------------------------------------------------------------------
# Request-level latency is not enough to operate a voice service. When p99
# regresses, the only question that matters is *which stage* — STT queueing,
# the agent, or TTS. These are labelled by stage so that question is one
# PromQL query, not a bisect.
#
# Buckets are tighter than the HTTP ones: voice SLOs live in the 100ms-2s
# range, and the default 0.1-30s buckets have almost no resolution there.

STAGE_LATENCY = Histogram(
    "voice_stage_latency_seconds",
    "Per-stage latency (stt, agent, tts).",
    labelnames=("stage",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0),
    registry=REGISTRY,
)

TIME_TO_FIRST_TRANSCRIPT = Histogram(
    "voice_time_to_first_transcript_seconds",
    "Audio arrival -> first partial transcript. The 'is it listening?' metric.",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.7, 1.0, 2.0),
    registry=REGISTRY,
)

FINAL_TRANSCRIPT_LATENCY = Histogram(
    "voice_final_transcript_latency_seconds",
    "End of speech -> final transcript.",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0),
    registry=REGISTRY,
)

AGENT_FIRST_TOKEN = Histogram(
    "voice_agent_first_token_seconds",
    "Final transcript -> first agent token.",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
    registry=REGISTRY,
)

TTS_TIME_TO_FIRST_AUDIO = Histogram(
    "voice_tts_time_to_first_audio_seconds",
    "First agent token -> first synthesized audio chunk.",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0),
    registry=REGISTRY,
)

END_TO_END_LATENCY = Histogram(
    "voice_end_to_end_latency_seconds",
    "End of user speech -> first audio of the response. The number users feel.",
    buckets=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    registry=REGISTRY,
)

# --- saturation / concurrency ---

ACTIVE_SESSIONS = Gauge(
    "voice_active_sessions",
    "Currently open voice sessions.",
    registry=REGISTRY,
)

AUDIO_QUEUE_DEPTH = Gauge(
    "voice_audio_queue_depth",
    "Requests waiting in the inference queue. Rising depth = saturation.",
    registry=REGISTRY,
)

REQUESTS_IN_FLIGHT = Gauge(
    "voice_requests_in_flight",
    "Inference requests currently executing.",
    registry=REGISTRY,
)

INFERENCE_BATCH_SIZE = Histogram(
    "voice_inference_batch_size",
    "Actual batch size at execution. If this sits at 1 under load, batching "
    "is not working and max_wait_ms is too low.",
    buckets=(1, 2, 4, 8, 16, 32, 64),
    registry=REGISTRY,
)

INFERENCE_QUEUE_WAIT = Histogram(
    "voice_inference_queue_wait_seconds",
    "Time spent waiting in queue before inference — the latency cost of batching.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

# --- throughput / errors ---

AUDIO_SECONDS_PROCESSED = Counter(
    "voice_audio_seconds_processed_total",
    "Cumulative audio seconds transcribed. Numerator for real-time factor.",
    registry=REGISTRY,
)

TRANSCRIPTION_RTF = Histogram(
    "voice_transcription_realtime_factor",
    "Compute seconds per audio second. <1 means faster than real time.",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0),
    registry=REGISTRY,
)

SESSION_CANCELLATIONS = Counter(
    "voice_session_cancellations_total",
    "Generations cancelled, by cause (barge-in, client disconnect).",
    labelnames=("reason",),
    registry=REGISTRY,
)

DROPPED_AUDIO_CHUNKS = Counter(
    "voice_dropped_audio_chunks_total",
    "Audio chunks dropped, by reason (queue_full, invalid, closed).",
    labelnames=("reason",),
    registry=REGISTRY,
)

VOICE_ERRORS = Counter(
    "voice_errors_total",
    "Voice pipeline errors by stage and type.",
    labelnames=("stage", "type"),
    registry=REGISTRY,
)


def render_metrics() -> bytes:
    """Return the current metric snapshot in Prometheus exposition format."""
    return generate_latest(REGISTRY)
