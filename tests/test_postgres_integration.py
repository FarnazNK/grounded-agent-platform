from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from rag_agent.config import Settings
from rag_agent.db import get_session_factory, init_engine
from rag_agent.postgres_store import PostgresRAGStore
from rag_agent.service import RAGService


@pytest.mark.integration
def test_postgres_ingestion_and_query_round_trip():
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL not set")

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
