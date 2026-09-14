from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from rag_agent.auth import create_access_token, hash_password, verify_password
from rag_agent.cache import TTLCache
from rag_agent.config import Settings, get_settings
from rag_agent.embeddings import EmbeddingService, build_embedding_service
from rag_agent.errors import (
    AuthenticationRequiredError,
    AuthorizationFailedError,
    DocumentNotFoundError,
    DuplicateIngestionError,
    IngestionNotFoundError,
    MalformedLLMOutputError,
    RateLimitExceededError,
    RequestValidationError,
    TokenLimitExceededError,
    WorkspaceNotFoundError,
)
from rag_agent.guardrails import (
    DataExfiltrationDetector,
    PIIDetector,
    PIILeakDetector,
    PromptInjectionDetector,
    apply_guardrails,
)
from rag_agent.guardrails.base import GuardrailAction, GuardrailViolation
from rag_agent.guardrails.output_filters import SourceCitationChecker
from rag_agent.llm import LLMProvider, build_llm_provider
from rag_agent.observability import get_logger
from rag_agent.parser import chunk_text, parse_document_bytes
from rag_agent.retrieval import HybridRetriever
from rag_agent.schemas import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    IngestionJobRecord,
    IngestionStatus,
    LatencyBreakdown,
    MembershipRole,
    QueryResult,
    UserRecord,
    WorkspaceMembership,
    WorkspaceRecord,
)
from rag_agent.store import RAGStore

log = get_logger(__name__)
_DUMMY_PASSWORD_HASH = hash_password("rag-agent-dummy-password-not-a-real-account")


@dataclass
class AuthResult:
    access_token: str
    user: UserRecord


class SlidingWindowRateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._events: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.time()
        cutoff = now - 60
        with self._lock:
            events = [event for event in self._events.get(key, []) if event >= cutoff]
            if len(events) >= self._limit:
                raise RateLimitExceededError()
            events.append(now)
            self._events[key] = events


class RAGService:
    def __init__(
        self,
        store: RAGStore,
        *,
        settings: Settings | None = None,
        embedding_service: EmbeddingService | None = None,
        llm_provider: LLMProvider | None = None,
        retrieval_cache: TTLCache[str, QueryResult] | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.embeddings = embedding_service or build_embedding_service(self.settings)
        self.llm = llm_provider or build_llm_provider(self.settings)
        self.retriever = HybridRetriever(store, self.settings)
        self.retrieval_cache = retrieval_cache or TTLCache[str, QueryResult](
            self.settings.retrieval_cache_ttl_seconds
        )
        self.rate_limiter = rate_limiter or SlidingWindowRateLimiter(
            self.settings.rate_limit_requests_per_minute
        )
        self._now = now or (lambda: datetime.now(UTC))

    def bootstrap_admin(
        self,
        *,
        email: str,
        password: str,
        organization_slug: str,
        organization_name: str,
        workspace_slug: str,
        workspace_name: str,
    ) -> WorkspaceMembership:
        if not self.settings.enable_bootstrap_admin:
            raise AuthorizationFailedError("Bootstrap is disabled.")
        if self.store.user_count() > 0:
            raise AuthorizationFailedError(
                "Bootstrap is only allowed before the first user exists."
            )
        email = email.strip().casefold()
        if self.store.get_user_by_email(email):
            raise RequestValidationError("User already exists.")
        user = self.store.create_user(email, hash_password(password))
        org = self.store.create_organization(organization_slug, organization_name)
        workspace = self.store.create_workspace(org.id, workspace_slug, workspace_name)
        self.store.add_membership(user.id, org.id, workspace.id, MembershipRole.admin)
        return WorkspaceMembership(workspace=workspace, organization=org, role=MembershipRole.admin)

    def authenticate(self, email: str, password: str) -> AuthResult:
        user = self.store.get_user_by_email(email.strip().casefold())
        password_hash = user.password_hash if user is not None else _DUMMY_PASSWORD_HASH
        password_ok = verify_password(password, password_hash)
        if user is None or not user.is_active or not password_ok:
            raise AuthenticationRequiredError("Invalid email or password.")
        return AuthResult(access_token=create_access_token(user.id, self.settings), user=user)

    def get_user(self, user_id: str) -> UserRecord:
        user = self.store.get_user(user_id)
        if not user or not user.is_active:
            raise AuthenticationRequiredError("Authenticated user is unavailable.")
        return user

    def list_workspaces(self, user_id: str) -> list[WorkspaceMembership]:
        return self.store.list_user_workspaces(user_id)

    def _require_workspace_role(
        self, user_id: str, workspace_id: str, minimum: MembershipRole
    ) -> WorkspaceRecord:
        workspace = self.store.get_workspace(workspace_id)
        if not workspace:
            raise WorkspaceNotFoundError(workspace_id)
        membership = self.store.get_membership(user_id, workspace_id)
        if not membership:
            raise AuthorizationFailedError("You are not a member of this workspace.")
        order = {MembershipRole.viewer: 0, MembershipRole.editor: 1, MembershipRole.admin: 2}
        if order[membership.role] < order[minimum]:
            raise AuthorizationFailedError(f"{minimum.value} role is required for this operation.")
        return workspace

    def ingest_document(
        self,
        *,
        user_id: str,
        workspace_id: str,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        self._require_workspace_role(user_id, workspace_id, MembershipRole.editor)
        parsed = parse_document_bytes(filename, content_type, data, settings=self.settings)
        duplicate = self.store.find_active_document_by_sha(workspace_id, parsed.sha256)
        if duplicate:
            job = self.store.create_ingestion_job(
                IngestionJobRecord(
                    workspace_id=workspace_id,
                    requested_by_user_id=user_id,
                    document_id=duplicate.id,
                    operation="upload",
                    status=IngestionStatus.duplicate,
                    error_code="duplicate_ingestion",
                    error_message="An active document with the same content already exists.",