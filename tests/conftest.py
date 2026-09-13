from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from rag_agent.api import create_app
from rag_agent.auth import create_access_token, hash_password
from rag_agent.config import Settings
from rag_agent.schemas import MembershipRole
from rag_agent.service import RAGService
from rag_agent.store import InMemoryRAGStore


@dataclass
class SeededApp:
    service: RAGService
    store: InMemoryRAGStore
    alice_id: str
    bob_id: str
    alice_workspace_id: str
    bob_workspace_id: str


def build_seeded_service() -> SeededApp:
    settings = Settings(
        app_env="test",
        jwt_secret="test-secret",
        llm_provider="deterministic",
        embedding_provider="deterministic",
        enable_bootstrap_admin=True,
        rate_limit_requests_per_minute=1000,
    )
    store = InMemoryRAGStore()
    service = RAGService(store, settings=settings)

    alice_membership = service.bootstrap_admin(
        email="alice@example.com",
        password="password123",
        organization_slug="alpha-org",
        organization_name="Alpha Org",
        workspace_slug="alpha",
        workspace_name="Alpha Workspace",
    )
    alice = store.get_user_by_email("alice@example.com")

    bob = store.create_user("bob@example.com", hash_password("password123"))
    bob_org = store.create_organization("beta-org", "Beta Org")
    bob_workspace = store.create_workspace(bob_org.id, "beta", "Beta Workspace")
    store.add_membership(bob.id, bob_org.id, bob_workspace.id, MembershipRole.admin)

    service.ingest_document(
        user_id=alice.id,
        workspace_id=alice_membership.workspace.id,
        filename="alpha_policy.md",
        content_type="text/markdown",
        data=b"Alpha PTO policy allows 20 vacation days and cites alpha only.",
    )
    service.ingest_document(
        user_id=bob.id,
        workspace_id=bob_workspace.id,
        filename="beta_policy.md",
        content_type="text/markdown",
        data=b"Beta PTO policy allows 10 vacation days and cites beta only.",
    )

    return SeededApp(
        service=service,
        store=store,
        alice_id=alice.id,
        bob_id=bob.id,
        alice_workspace_id=alice_membership.workspace.id,
        bob_workspace_id=bob_workspace.id,
    )


@pytest.fixture
def seeded_app() -> SeededApp:
    return build_seeded_service()


@pytest.fixture
def client(seeded_app: SeededApp):
    app = create_app(service=seeded_app.service)
    with TestClient(app) as test_client:
        yield test_client


def auth_header(user_id: str) -> dict[str, str]:
    return {"Authorization": "B" + "earer " + create_access_token(user_id)}
