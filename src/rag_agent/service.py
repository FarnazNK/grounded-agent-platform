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
                    completed_at=self._now(),
                )
            )
            raise DuplicateIngestionError(
                "Duplicate ingestion request rejected.",
                details={"document_id": duplicate.id, "job_id": job.id},
            )

        document = self.store.create_document(
            DocumentRecord(
                workspace_id=workspace_id,
                created_by_user_id=user_id,
                source_name=parsed.source_name,
                media_type=parsed.media_type,
                file_size=parsed.file_size,
                sha256=parsed.sha256,
                status=DocumentStatus.pending,
                content_text=parsed.text,
                metadata=parsed.metadata,
            )
        )
        job = self.store.create_ingestion_job(
            IngestionJobRecord(
                workspace_id=workspace_id,
                requested_by_user_id=user_id,
                document_id=document.id,
                operation="upload",
                status=IngestionStatus.pending,
            )
        )
        return self._index_document(document=document, job=job)

    def _index_document(
        self, *, document: DocumentRecord, job: IngestionJobRecord
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        self.store.update_ingestion_job(
            job.id, status=IngestionStatus.processing, attempt_count=job.attempt_count + 1
        )
        self.store.update_document(document.id, status=DocumentStatus.processing)
        try:
            chunks = chunk_text(document.content_text, settings=self.settings)
            embeddings = self.embeddings.embed_documents([chunk.content for chunk in chunks])
            records = [
                ChunkRecord(
                    document_id=document.id,
                    workspace_id=document.workspace_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    embedding=embedding,
                    metadata=chunk.metadata,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
            self.store.replace_document_chunks(
                document.id, document.workspace_id, records, self.settings.embedding_model
            )
            document = self.store.update_document(
                document.id,
                status=DocumentStatus.ready,
                metadata={**document.metadata, "chunk_count": len(records)},
                error_code=None,
                error_message=None,
            )
            self.store.bump_workspace_index_version(document.workspace_id)
            self.retrieval_cache.invalidate_prefix(f"{document.workspace_id}:")
            job = self.store.update_ingestion_job(
                job.id,
                status=IngestionStatus.completed,
                completed_at=self._now(),
                error_code=None,
                error_message=None,
            )
            return document, job
        except Exception as exc:
            self.store.delete_document_chunks(document.id)
            self.store.update_document(
                document.id,
                status=DocumentStatus.failed,
                error_code=getattr(exc, "code", "ingestion_failed"),
                error_message=str(exc),
            )
            self.store.update_ingestion_job(
                job.id,
                status=IngestionStatus.failed,
                completed_at=self._now(),
                error_code=getattr(exc, "code", "ingestion_failed"),
                error_message=str(exc),
            )
            log.warning("ingestion.failed", document_id=document.id, error=str(exc))
            raise

    def reindex_document(
        self, *, user_id: str, workspace_id: str, document_id: str
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        self._require_workspace_role(user_id, workspace_id, MembershipRole.editor)
        document = self.store.get_document(document_id, workspace_id)
        if not document or document.status == DocumentStatus.deleted:
            raise DocumentNotFoundError(document_id)
        job = self.store.create_ingestion_job(
            IngestionJobRecord(
                workspace_id=workspace_id,
                requested_by_user_id=user_id,
                document_id=document_id,
                operation="reindex",
                status=IngestionStatus.pending,
            )
        )
        return self._index_document(document=document, job=job)

    def delete_document(
        self, *, user_id: str, workspace_id: str, document_id: str
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        self._require_workspace_role(user_id, workspace_id, MembershipRole.editor)
        document = self.store.get_document(document_id, workspace_id)
        if not document:
            raise DocumentNotFoundError(document_id)
        self.store.delete_document_chunks(document_id)
        document = self.store.update_document(document_id, status=DocumentStatus.deleted)
        self.store.bump_workspace_index_version(workspace_id)
        self.retrieval_cache.invalidate_prefix(f"{workspace_id}:")
        job = self.store.create_ingestion_job(
            IngestionJobRecord(
                workspace_id=workspace_id,
                requested_by_user_id=user_id,
                document_id=document_id,
                operation="delete",
                status=IngestionStatus.completed,
                completed_at=self._now(),
            )
        )
        return document, job

    def get_document(self, *, user_id: str, workspace_id: str, document_id: str) -> DocumentRecord:
        self._require_workspace_role(user_id, workspace_id, MembershipRole.viewer)
        document = self.store.get_document(document_id, workspace_id)
        if not document:
            raise DocumentNotFoundError(document_id)
        return document

    def list_documents(self, *, user_id: str, workspace_id: str) -> list[DocumentRecord]:
        self._require_workspace_role(user_id, workspace_id, MembershipRole.viewer)
        return self.store.list_documents(workspace_id)

    def get_ingestion_job(
        self, *, user_id: str, workspace_id: str, job_id: str
    ) -> IngestionJobRecord:
        self._require_workspace_role(user_id, workspace_id, MembershipRole.viewer)
        job = self.store.get_ingestion_job(job_id)
        if not job or job.workspace_id != workspace_id:
            raise IngestionNotFoundError(job_id)
        return job

    def query(self, *, user_id: str, workspace_id: str, query: str, request_id: str) -> QueryResult:
        workspace = self._require_workspace_role(user_id, workspace_id, MembershipRole.viewer)
        self.rate_limiter.check(f"{user_id}:{workspace_id}:query")
        if len(query.split()) * 4 > self.settings.llm_max_prompt_tokens:
            raise TokenLimitExceededError()
        try:
            sanitized_query, _ = apply_guardrails(
                query,
                [PIIDetector(), PromptInjectionDetector(), DataExfiltrationDetector()],
            )
        except GuardrailViolation as exc:
            if exc.decision.action == GuardrailAction.BLOCK:
                return QueryResult(
                    request_id=request_id,
                    answer="Request blocked by guardrail.",
                    refusal=True,
                    grounded=True,
                )
            raise
        cache_key = (
            f"{workspace.id}:{workspace.index_version}:"
            f"{sanitized_query}:{self.settings.top_k_final}"
        )
        cached = self.retrieval_cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"request_id": request_id, "cached": True})

        start = time.perf_counter()
        query_embedding = self.embeddings.embed_query(sanitized_query)
        chunks, retrieval_metrics = self.retriever.retrieve(
            workspace_id, sanitized_query, query_embedding
        )
        safe_chunks = []
        injection_detector = PromptInjectionDetector()
        for chunk in chunks:
            decision = injection_detector(chunk.content)
            if decision.action != GuardrailAction.BLOCK:
                safe_chunks.append(chunk)
        chunks = safe_chunks
        llm_result = self.llm.answer_question(sanitized_query, chunks)
        output_text, _ = apply_guardrails(
            llm_result.text,
            [
                PIILeakDetector(),
                SourceCitationChecker(
                    valid_sources=frozenset(chunk.source_name for chunk in chunks)
                ),
            ],
            raise_on_block=False,
        )
        if not output_text.strip():
            raise MalformedLLMOutputError()
        citations = sorted(set(re.findall(r"\[source:\s*([^\]]+)\]", output_text, flags=re.I)))
        result = QueryResult(
            request_id=request_id,
            answer=output_text,
            citations=citations,
            chunks=chunks,
            usage=llm_result.usage,
            latency=LatencyBreakdown(
                total_ms=(time.perf_counter() - start) * 1000,
                retrieval_ms=retrieval_metrics["dense_ms"] + retrieval_metrics["sparse_ms"],
                llm_ms=llm_result.latency_ms,
                db_ms=retrieval_metrics["db_ms"],
            ),
            cached=False,
            grounded=(
                (not chunks and not citations)
                or (
                    bool(citations)
                    and set(citations).issubset({chunk.source_name for chunk in chunks})
                )
            ),
            sanitized_query=sanitized_query if sanitized_query != query else None,
            refusal=False,
        )
        self.retrieval_cache.set(cache_key, result)
        return result
