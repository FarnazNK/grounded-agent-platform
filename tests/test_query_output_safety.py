from __future__ import annotations

from rag_agent.config import Settings
from rag_agent.llm import LLMResult
from rag_agent.schemas import UsageInfo
from rag_agent.service import RAGService
from rag_agent.store import InMemoryRAGStore

DOC = (
    b"PTO policy: new hires receive 15 PTO days per year. "
    b"Questions go to hr-lead@example.com or 555-123-4567."
)


class ScriptedLLM:
    def __init__(self, text: str) -> None:
        self.text = text

    def answer_question(self, query, chunks):
        return LLMResult(text=self.text, usage=UsageInfo(), latency_ms=0.0)


def query_with(answer: str):
    settings = Settings(
        app_env="test",
        jwt_secret="output-safety-secret-long-enough-123456",
        enable_bootstrap_admin=True,
        rate_limit_requests_per_minute=1000,
    )
    store = InMemoryRAGStore()
    service = RAGService(store, settings=settings, llm_provider=ScriptedLLM(answer))
    membership = service.bootstrap_admin(
        email="admin@example.com",
        password="password123",
        organization_slug="org",
        organization_name="Org",
        workspace_slug="default",
        workspace_name="Default",
    )
    user = store.get_user_by_email("admin@example.com")
    assert user is not None
    workspace_id = membership.workspace.id
    service.ingest_document(
        user_id=user.id,
        workspace_id=workspace_id,
        filename="pto.md",
        content_type="text/markdown",
        data=DOC,
    )
    return service.query(
        user_id=user.id,
        workspace_id=workspace_id,
        query="How many PTO days do new hires get?",
        request_id="req-1",
    )


def test_returned_chunks_are_redacted_like_the_answer():
    result = query_with("New hires get 15 days [source: pto.md].")

    assert result.chunks, "expected retrieved chunks"
    returned = " ".join(chunk.content for chunk in result.chunks)
    assert "hr-lead@example.com" not in returned
    assert "555-123-4567" not in returned
    assert "[REDACTED_EMAIL]" in returned


def test_cited_answer_is_grounded():
    result = query_with("New hires get 15 days [source: pto.md].")

    assert result.grounded is True
    assert result.grounding_issue is None
    assert result.citations == ["pto.md"]


def test_answer_without_citation_is_marked_ungrounded():
    result = query_with("New hires get 90 days.")

    assert result.grounded is False
    assert result.grounding_issue == "missing_citation"


def test_fabricated_citation_is_removed_and_marked_ungrounded():
    result = query_with("New hires get 90 days [source: made-up.md].")

    assert "made-up.md" not in result.answer
    assert "unverified" not in result.answer
    assert result.citations == []
    assert result.grounded is False
    assert result.grounding_issue == "invalid_citation"


def test_nested_chunk_metadata_is_redacted():
    from rag_agent.schemas import RetrievedChunk
    from rag_agent.service import _redact_chunks

    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        source_name="pto.md",
        content="contact hr-lead@example.com",
        fused_score=1.0,
        metadata={
            "author": {"email": "owner@example.com", "phones": ["555-123-4567"]},
            "pages": [1, 2],
            "tags": ("cc 555-987-6543",),
        },
    )

    [redacted] = _redact_chunks([chunk])

    dumped = redacted.model_dump_json()
    for secret in ("hr-lead@example.com", "owner@example.com", "555-123-4567", "555-987-6543"):
        assert secret not in dumped
    assert redacted.metadata["pages"] == [1, 2]
    assert isinstance(redacted.metadata["tags"], tuple)
