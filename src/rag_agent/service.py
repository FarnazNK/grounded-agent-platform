from __future__ import annotations

import base64
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, NoReturn

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
from rag_agent.evidence import REFUSAL_ANSWER, assess_evidence, topic_terms
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
    RetrievedChunk,
    UserRecord,
    WorkspaceMembership,
    WorkspaceRecord,
)
from rag_agent.store import DocumentCursor, DuplicateDocumentConflict, RAGStore
from rag_agent.tracing import get_tracer

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
        membership = self.store.bootstrap_first_admin(
            email=email.strip().casefold(),
            password_hash=hash_password(password),
            organization_slug=organization_slug,
            organization_name=organization_name,
            workspace_slug=workspace_slug,
            workspace_name=workspace_name,
        )
        if membership is None:
            raise AuthorizationFailedError(
                "Bootstrap is only allowed before the first user exists."
            )
        return membership

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
        # Fast path: skip chunking and embedding for an obvious re-upload. The database's
        # unique index is what actually guarantees no duplicates (see publish below).
        duplicate = self.store.find_active_document_by_sha(workspace_id, parsed.sha256)
        if duplicate:
            self._reject_duplicate(user_id, workspace_id, duplicate.id)

        document = DocumentRecord(
            workspace_id=workspace_id,
            created_by_user_id=user_id,
            source_name=parsed.source_name,
            media_type=parsed.media_type,
            file_size=parsed.file_size,
            sha256=parsed.sha256,
            status=DocumentStatus.ready,
            content_text=parsed.text,
            metadata=parsed.metadata,
        )
        # Embedding can call a slow external API, so it runs before the transaction opens;
        # the database only sees one short write of the finished document.
        try:
            chunks = self._build_chunks(document)
        except Exception as exc:
            code = getattr(exc, "code", "ingestion_failed")
            self.store.record_failed_document(
                document.model_copy(
                    update={
                        "status": DocumentStatus.failed,
                        "error_code": code,
                        "error_message": str(exc),
                    }
                ),
                self._job(
                    document,
                    user_id,
                    "upload",
                    IngestionStatus.failed,
                    error_code=code,
                    error_message=str(exc),
                ),
            )
            log.warning("ingestion.failed", document_id=document.id, error=str(exc))
            raise

        document = document.model_copy(
            update={"metadata": {**document.metadata, "chunk_count": len(chunks)}}
        )
        try:
            document, job = self.store.publish_document(
                document,
                chunks,
                self._job(document, user_id, "upload", IngestionStatus.completed),
                self.settings.embedding_model,
            )
        except DuplicateDocumentConflict as conflict:
            # Lost a race with a concurrent upload of the same file.
            self._reject_duplicate(user_id, workspace_id, conflict.existing_document_id)
        self.retrieval_cache.invalidate_prefix(f"{workspace_id}:")
        return document, job

    def _reject_duplicate(self, user_id: str, workspace_id: str, existing_id: str) -> NoReturn:
        job = self.store.create_ingestion_job(
            IngestionJobRecord(
                workspace_id=workspace_id,
                requested_by_user_id=user_id,
                document_id=existing_id or None,
                operation="upload",
                status=IngestionStatus.duplicate,
                error_code="duplicate_ingestion",
                error_message="An active document with the same content already exists.",
                completed_at=self._now(),
            )
        )
        raise DuplicateIngestionError(
            "Duplicate ingestion request rejected.",
            details={"document_id": existing_id, "job_id": job.id},
        )

    def _build_chunks(self, document: DocumentRecord) -> list[ChunkRecord]:
        chunks = chunk_text(document.content_text, settings=self.settings)
        embeddings = self.embeddings.embed_documents([chunk.content for chunk in chunks])
        return [
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

    def _job(
        self,
        document: DocumentRecord,
        user_id: str,
        operation: str,
        status: IngestionStatus,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> IngestionJobRecord:
        return IngestionJobRecord(
            workspace_id=document.workspace_id,
            requested_by_user_id=user_id,
            document_id=document.id,
            operation=operation,
            status=status,
            attempt_count=1,
            error_code=error_code,
            error_message=error_message,
            completed_at=self._now(),
        )

    def reindex_document(
        self, *, user_id: str, workspace_id: str, document_id: str
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        self._require_workspace_role(user_id, workspace_id, MembershipRole.editor)
        document = self.store.get_document(document_id, workspace_id)
        if not document or document.status == DocumentStatus.deleted:
            raise DocumentNotFoundError(document_id)
        try:
            chunks = self._build_chunks(document)
        except Exception as exc:
            # The previous chunks stay in place, so search keeps working on the old index.
            self.store.create_ingestion_job(
                self._job(
                    document,
                    user_id,
                    "reindex",
                    IngestionStatus.failed,
                    error_code=getattr(exc, "code", "ingestion_failed"),
                    error_message=str(exc),
                )
            )
            log.warning("reindex.failed", document_id=document.id, error=str(exc))
            raise
        # Raises DocumentNotFoundError if the document was deleted while we were embedding.
        document, job = self.store.replace_document_index(
            document_id,
            workspace_id,
            chunks,
            self._job(document, user_id, "reindex", IngestionStatus.completed),
            self.settings.embedding_model,
            {**document.metadata, "chunk_count": len(chunks)},
        )
        self.retrieval_cache.invalidate_prefix(f"{workspace_id}:")
        return document, job

    def delete_document(
        self, *, user_id: str, workspace_id: str, document_id: str
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        self._require_workspace_role(user_id, workspace_id, MembershipRole.editor)
        document = self.store.get_document(document_id, workspace_id)
        if not document:
            raise DocumentNotFoundError(document_id)
        document, job = self.store.delete_document(
            document_id,
            workspace_id,
            self._job(document, user_id, "delete", IngestionStatus.completed),
        )
        self.retrieval_cache.invalidate_prefix(f"{workspace_id}:")
        return document, job

    def get_document(self, *, user_id: str, workspace_id: str, document_id: str) -> DocumentRecord:
        self._require_workspace_role(user_id, workspace_id, MembershipRole.viewer)
        document = self.store.get_document(document_id, workspace_id)
        if not document:
            raise DocumentNotFoundError(document_id)
        return document

    def list_documents(
        self, *, user_id: str, workspace_id: str, limit: int = 50, cursor: str | None = None
    ) -> tuple[list[DocumentRecord], str | None]:
        """One page of documents, oldest first, and the cursor for the next page (or None)."""
        self._require_workspace_role(user_id, workspace_id, MembershipRole.viewer)
        after = decode_document_cursor(cursor) if cursor else None
        # Fetch one extra row to learn whether another page exists.
        documents = self.store.list_documents(workspace_id, limit=limit + 1, after=after)
        if len(documents) <= limit:
            return documents, None
        page = documents[:limit]
        return page, encode_document_cursor(page[-1])

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

        tracer = get_tracer()
        with (
            tracer.trace_attributes(
                trace_name="rag.query",
                user_id=user_id,
                session_id=request_id,
                tags=["rag"],
                metadata={"workspace_id": workspace_id},
            ),
            tracer.observe("rag.query", kind="chain", input={"query": sanitized_query}) as span,
        ):
            result = self._answer_uncached(
                workspace_id=workspace_id,
                query=query,
                sanitized_query=sanitized_query,
                request_id=request_id,
            )
            span.update(
                output={"answer": result.answer, "citations": result.citations},
                metadata={
                    "grounded": result.grounded,
                    "chunks": len(result.chunks),
                    "total_ms": round(result.latency.total_ms, 2),
                },
            )
            tracer.score("grounded", 1.0 if result.grounded else 0.0)
        self.retrieval_cache.set(cache_key, result)
        return result

    def _answer_uncached(
        self, *, workspace_id: str, query: str, sanitized_query: str, request_id: str
    ) -> QueryResult:
        tracer = get_tracer()
        start = time.perf_counter()
        query_embedding = self.embeddings.embed_query(sanitized_query)
        with tracer.observe(
            "retrieve", kind="retriever", input={"query": sanitized_query}
        ) as retrieval_span:
            chunks, retrieval_metrics = self.retriever.retrieve(
                workspace_id, sanitized_query, query_embedding
            )
            retrieval_span.update(
                output=[chunk.source_name for chunk in chunks],
                metadata={key: round(value, 2) for key, value in retrieval_metrics.items()},
            )
        safe_chunks = []
        injection_detector = PromptInjectionDetector()
        for chunk in chunks:
            decision = injection_detector(chunk.content)
            if decision.action != GuardrailAction.BLOCK:
                safe_chunks.append(chunk)
        chunks = safe_chunks
        if self.settings.min_evidence_coverage > 0:
            frequencies, total_chunks = self.store.term_document_frequencies(
                workspace_id, topic_terms(sanitized_query)
            )
            evidence = assess_evidence(sanitized_query, chunks, frequencies, total_chunks)
            if not chunks or evidence.coverage < self.settings.min_evidence_coverage:
                log.info(
                    "query_refused_insufficient_evidence",
                    request_id=request_id,
                    coverage=round(evidence.coverage, 3),
                    missing_terms=evidence.missing_terms,
                )
                tracer.score("evidence_coverage", evidence.coverage)
                return QueryResult(
                    request_id=request_id,
                    answer=REFUSAL_ANSWER,
                    latency=LatencyBreakdown(
                        total_ms=(time.perf_counter() - start) * 1000,
                        retrieval_ms=retrieval_metrics["dense_ms"] + retrieval_metrics["sparse_ms"],
                        db_ms=retrieval_metrics["db_ms"],
                    ),
                    grounded=True,
                    sanitized_query=sanitized_query if sanitized_query != query else None,
                    refusal=True,
                )
        llm_result = self.llm.answer_question(sanitized_query, chunks)
        output_text, decisions = apply_guardrails(
            llm_result.text,
            [
                PIILeakDetector(),
                SourceCitationChecker(
                    valid_sources=frozenset(chunk.source_name for chunk in chunks),
                    require_citation=True,
                ),
            ],
            raise_on_block=False,
        )
        if not output_text.strip():
            raise MalformedLLMOutputError()
        citations = sorted(set(re.findall(r"\[source:\s*([^\]]+)\]", output_text, flags=re.I)))
        citation_decision = decisions[-1]
        grounding_issue: Literal["missing_citation", "invalid_citation"] | None = None
        if citation_decision.action == GuardrailAction.SANITIZE:
            grounding_issue = "invalid_citation"
        elif citation_decision.action == GuardrailAction.FLAG:
            grounding_issue = "missing_citation"
        result = QueryResult(
            request_id=request_id,
            answer=output_text,
            citations=citations,
            chunks=_redact_chunks(chunks),
            usage=llm_result.usage,
            latency=LatencyBreakdown(
                total_ms=(time.perf_counter() - start) * 1000,
                retrieval_ms=retrieval_metrics["dense_ms"] + retrieval_metrics["sparse_ms"],
                llm_ms=llm_result.latency_ms,
                db_ms=retrieval_metrics["db_ms"],
            ),
            cached=False,
            grounded=grounding_issue is None and (bool(citations) or not chunks),
            grounding_issue=grounding_issue,
            sanitized_query=sanitized_query if sanitized_query != query else None,
            refusal=False,
        )
        return result


def _redact_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Apply the answer's PII redaction to everything returned with the source chunks."""
    redactor = PIILeakDetector()

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            decision = redactor(value)
            return decision.sanitized_text if decision.sanitized_text is not None else value
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(redact(item) for item in value)
        return value

    return [
        chunk.model_copy(
            update={"content": redact(chunk.content), "metadata": redact(chunk.metadata)}
        )
        for chunk in chunks
    ]


def encode_document_cursor(document: DocumentRecord) -> str:
    """Opaque page cursor holding the last document's (created_at, id) sort key."""
    raw = json.dumps({"created_at": document.created_at.isoformat(), "id": document.id})
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_document_cursor(cursor: str) -> DocumentCursor:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode()))
        return DocumentCursor(
            created_at=datetime.fromisoformat(data["created_at"]), id=str(data["id"])
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise RequestValidationError("Invalid pagination cursor.") from exc
