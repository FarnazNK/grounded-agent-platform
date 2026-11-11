from __future__ import annotations

from contextlib import contextmanager

import pytest

from rag_agent.config import Settings
from rag_agent.tracing import LangfuseTracer, Tracer, build_tracer, clip


def test_tracing_is_disabled_by_default():
    tracer = build_tracer(Settings())

    assert type(tracer) is Tracer
    assert tracer.enabled is False
    assert tracer.langchain_callbacks() == []
    with tracer.observe("x", input={"a": 1}) as span:
        span.update(output="ignored")


def test_langfuse_backend_without_credentials_falls_back_to_noop(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    tracer = build_tracer(Settings(tracing_backend="langfuse"))

    assert tracer.enabled is False


def test_clip_bounds_exported_payloads():
    clipped = clip({"content": "x" * 10_000, "items": list(range(100))})

    assert len(clipped["content"]) < 4_100
    assert clipped["content"].endswith("[10000 chars]")
    assert len(clipped["items"]) == 50


class FakeSpan:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, **values):
        self.updates.append(values)


class FakeManager:
    def __init__(self, span: FakeSpan, log: list) -> None:
        self.span = span
        self.log = log

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc, tb):
        self.log.append(exc_type)
        return False


class FakeClient:
    def __init__(self, *, fail_start: bool = False) -> None:
        self.fail_start = fail_start
        self.exits: list = []
        self.span = FakeSpan()
        self.started: list[dict] = []

    def start_as_current_observation(self, **kwargs):
        if self.fail_start:
            raise RuntimeError("backend down")
        self.started.append(kwargs)
        return FakeManager(self.span, self.exits)


def make_tracer(client: FakeClient) -> LangfuseTracer:
    tracer = LangfuseTracer.__new__(LangfuseTracer)
    tracer._client = client

    @contextmanager
    def propagate(**_kwargs):
        yield

    tracer._propagate_attributes = propagate
    return tracer


def test_langfuse_tracer_records_spans_and_clips_input():
    client = FakeClient()
    tracer = make_tracer(client)

    with tracer.observe("agent.coder", kind="agent", input={"big": "y" * 9_000}) as span:
        span.update(output="done")

    assert client.started[0]["as_type"] == "agent"
    assert len(client.started[0]["input"]["big"]) < 4_100
    assert client.span.updates == [{"output": "done"}]
    assert client.exits == [None]


def test_langfuse_tracer_propagates_application_errors():
    client = FakeClient()
    tracer = make_tracer(client)

    with pytest.raises(ValueError), tracer.observe("tool.read_file"):
        raise ValueError("tool failed")

    assert client.exits == [ValueError]


def test_tracing_backend_failure_never_breaks_the_caller():
    tracer = make_tracer(FakeClient(fail_start=True))

    with tracer.trace_attributes(session_id="s1"), tracer.observe("rag.query") as span:
        span.update(output="still works")
        value = 42

    assert value == 42
