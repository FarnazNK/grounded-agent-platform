from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from rag_agent.config import Settings
from rag_agent.errors import AuthorizationFailedError
from rag_agent.service import RAGService
from rag_agent.store import InMemoryRAGStore


def _settings(**overrides) -> Settings:
    return Settings(
        app_env="test",
        jwt_secret="bootstrap-secret-which-is-long-enough-123",
        enable_bootstrap_admin=True,
        **overrides,
    )


def _race(service: RAGService, attempts: int = 8) -> list[object]:
    def attempt(index: int):
        try:
            return service.bootstrap_admin(
                email=f"admin{index}@example.com",
                password="password123",
                organization_slug=f"org-{index}",
                organization_name="Org",
                workspace_slug="default",
                workspace_name="Default",
            )
        except AuthorizationFailedError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        return list(pool.map(attempt, range(attempts)))


def test_concurrent_bootstrap_creates_exactly_one_admin_in_memory():
    store = InMemoryRAGStore()
    results = _race(RAGService(store, settings=_settings()))

    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert store.user_count() == 1
    assert len(store.organizations) == 1
    assert len(store.memberships) == 1


@pytest.mark.integration
def test_concurrent_bootstrap_creates_exactly_one_admin_in_postgres():
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL not set")

    from rag_agent.db import get_session_factory, init_engine
    from rag_agent.postgres_store import PostgresRAGStore

    engine = init_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE memberships, workspaces, organizations, users CASCADE"))

    store = PostgresRAGStore(get_session_factory())
    results = _race(RAGService(store, settings=_settings(database_url=database_url)))

    assert sum(not isinstance(r, Exception) for r in results) == 1
    with engine.connect() as conn:
        for table in ("users", "organizations", "workspaces", "memberships"):
            assert conn.scalar(text(f"SELECT count(*) FROM {table}")) == 1
