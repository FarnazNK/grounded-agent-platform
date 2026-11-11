from __future__ import annotations

import pytest
from sqlalchemy import text

from rag_agent.config import Settings
from rag_agent.db import get_session_factory, init_engine
from rag_agent.service import RAGService
from tests.conftest import disposable_database_url


@pytest.mark.integration
def test_postgres_ingestion_and_query_round_trip():
    database_url = disposable_database_url()

    from rag_agent.postgres_store import PostgresRAGStore

    engine = init_engine(database_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE ingestion_jobs, document_chunks, documents, memberships, "
                "workspaces, organizations, users RESTART IDENTITY CASCADE"
            )
        )

    settings = Settings(
        app_env="test",
        database_url=database_url,
        jwt_secret="integration-secret-which-is-long-enough-123",
        llm_provider="deterministic",
        embedding_provider="deterministic",
        enable_bootstrap_admin=True,
    )
    init_engine(database_url)
    service = RAGService(PostgresRAGStore(get_session_factory()), settings=settings)
    membership = service.bootstrap_admin(
        email="postgres@example.com",
        password="password123",
        organization_slug="postgres-org",
        organization_name="Postgres Org",
        workspace_slug="integration",
        workspace_name="Integration Workspace",
    )
    user = service.store.get_user_by_email("postgres@example.com")
    assert user is not None

    document, job = service.ingest_document(
        user_id=user.id,
        workspace_id=membership.workspace.id,
        filename="postgres.md",
        content_type="text/markdown",
        data=b"PostgreSQL backed retrieval uses pgvector and tenant scoped chunks.",
    )
    assert document.status == "ready"
    assert job.status == "completed"

    result = service.query(
        user_id=user.id,
        workspace_id=membership.workspace.id,
        query="What backs retrieval?",
        request_id="integration",
    )
    assert result.chunks
    assert result.chunks[0].source_name == "postgres.md"
    assert isinstance(result.chunks[0].chunk_id, str)
    assert isinstance(result.chunks[0].document_id, str)
    assert result.chunks[0].document_id == document.id

    # Same matching rules as the in-memory store: stems match short suffixes ("back" ~
    # "backed"), short stems need an exact word, and "pg" does not match inside "pgvector".
    stems = ["back", "tenant", "weather", "pg", "post"]
    frequencies, total = service.store.term_document_frequencies(membership.workspace.id, stems)
    assert total == 1
    assert frequencies == {"back": 1, "tenant": 1, "weather": 0, "pg": 0, "post": 0}
    from rag_agent.store import InMemoryRAGStore

    memory = InMemoryRAGStore()
    memory_service = RAGService(memory, settings=settings.model_copy(update={"database_url": None}))
    memory_membership = memory_service.bootstrap_admin(
        email="m@example.com",
        password="password123",
        organization_slug="m",
        organization_name="M",
        workspace_slug="m",
        workspace_name="M",
    )
    memory_user = memory.get_user_by_email("m@example.com")
    assert memory_user is not None
    memory_service.ingest_document(
        user_id=memory_user.id,
        workspace_id=memory_membership.workspace.id,
        filename="postgres.md",
        content_type="text/markdown",
        data=b"PostgreSQL backed retrieval uses pgvector and tenant scoped chunks.",
    )
    assert memory.term_document_frequencies(memory_membership.workspace.id, stems) == (
        frequencies,
        total,
    )

    refused = service.query(
        user_id=user.id,
        workspace_id=membership.workspace.id,
        query="What's the weather in Tokyo tomorrow?",
        request_id="integration-refusal",
    )
    assert refused.refusal is True
    assert refused.chunks == []
