from __future__ import annotations

import time
from datetime import UTC, datetime

from sqlalchemy import TextClause, delete, func, literal, select, text, tuple_, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from rag_agent.errors import DatabaseUnavailableError, DocumentNotFoundError
from rag_agent.models import (
    Document,
    DocumentChunk,
    IngestionJob,
    Membership,
    Organization,
    User,
    Workspace,
)
from rag_agent.schemas import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    IngestionJobRecord,
    MembershipRecord,
    MembershipRole,
    OrganizationRecord,
    RetrievedChunk,
    UserRecord,
    WorkspaceMembership,
    WorkspaceRecord,
)
from rag_agent.store import (
    STEM_MAX_EXTRA,
    DocumentCursor,
    DuplicateDocumentConflict,
)

# Arbitrary constant identifying the bootstrap critical section.
_BOOTSTRAP_LOCK_KEY = 7_041_202_611

# Matches the predicate of the partial unique index, so lookups can use it.
_ACTIVE_SHA_INDEX = "uq_documents_workspace_sha_active"
_ACTIVE_DOCUMENT_PREDICATE = Document.status.not_in(
    (DocumentStatus.deleted.value, DocumentStatus.failed.value)
)


def _constraint_name(exc: IntegrityError) -> str | None:
    diag = getattr(exc.orig, "diag", None)
    return getattr(diag, "constraint_name", None)


def _dense_sql(order_by: str) -> TextClause:
    return text(
        f"""
        WITH nearest AS MATERIALIZED (
            SELECT c.id, c.document_id, c.content, c.chunk_metadata,
                   c.embedding <=> CAST(:embedding AS vector) AS distance
            FROM document_chunks c
            WHERE c.workspace_id = :workspace_id
            ORDER BY {order_by}
            LIMIT :limit
        )
        SELECT n.id, n.document_id, d.source_name, n.content, n.chunk_metadata,
               1 - n.distance AS score
        FROM nearest n
        JOIN documents d ON d.id = n.document_id
        WHERE d.status = 'ready'
        ORDER BY n.distance
        """
    )


# Ordering by the bare distance operator lets the planner use the HNSW index.
_ANN_DENSE_SQL = _dense_sql("c.embedding <=> CAST(:embedding AS vector)")
# "+ 0" makes the sort key an expression the HNSW index cannot serve, so the planner reads
# the workspace's rows through ix_document_chunks_workspace_document and sorts them: exact.
_EXACT_DENSE_SQL = _dense_sql("(c.embedding <=> CAST(:embedding AS vector)) + 0")


class PostgresRAGStore:
    def __init__(
        self, session_factory: sessionmaker[Session], *, exact_search_max_chunks: int = 20_000
    ) -> None:
        self._session_factory = session_factory
        self._exact_search_max_chunks = exact_search_max_chunks
        self._pgvector_version: tuple[int, ...] | None = None

    def _session(self) -> Session:
        return self._session_factory()

    def _run(self, fn):
        try:
            return fn()
        except OperationalError as exc:
            raise DatabaseUnavailableError() from exc

    def ping(self) -> None:
        def inner() -> None:
            with self._session() as session:
                session.execute(text("SELECT 1"))

        self._run(inner)

    def user_count(self) -> int:
        def inner():
            with self._session() as session:
                return int(session.scalar(select(func.count()).select_from(User)) or 0)

        return self._run(inner)

    def bootstrap_first_admin(
        self,
        *,
        email: str,
        password_hash: str,
        organization_slug: str,
        organization_name: str,
        workspace_slug: str,
        workspace_name: str,
    ) -> WorkspaceMembership | None:
        """Create the first admin in one transaction.

        A transaction-scoped advisory lock serializes concurrent bootstrap calls, so only
        the first caller sees zero users; a failure part-way rolls everything back.
        """

        def inner():
            with self._session() as session, session.begin():
                session.execute(
                    text("SELECT pg_advisory_xact_lock(:key)"), {"key": _BOOTSTRAP_LOCK_KEY}
                )
                if session.scalar(select(func.count()).select_from(User)):
                    return None
                user = User(email=email, password_hash=password_hash, is_active=True)
                org = Organization(slug=organization_slug, name=organization_name)
                session.add_all([user, org])
                session.flush()
                workspace = Workspace(
                    organization_id=org.id,
                    slug=workspace_slug,
                    name=workspace_name,
                    index_version=1,
                )
                session.add(workspace)
                session.flush()
                session.add(
                    Membership(
                        user_id=user.id,
                        organization_id=org.id,
                        workspace_id=workspace.id,
                        role=MembershipRole.admin.value,
                    )
                )
                session.flush()
                return WorkspaceMembership(
                    workspace=WorkspaceRecord.model_validate(workspace),
                    organization=OrganizationRecord.model_validate(org),
                    role=MembershipRole.admin,
                )

        return self._run(inner)

    def create_user(self, email: str, password_hash: str) -> UserRecord:
        def inner():
            with self._session() as session:
                user = User(email=email, password_hash=password_hash, is_active=True)
                session.add(user)
                session.commit()
                session.refresh(user)
                return UserRecord.model_validate(user)

        return self._run(inner)

    def get_user(self, user_id: str) -> UserRecord | None:
        def inner():
            with self._session() as session:
                user = session.get(User, user_id)
                return UserRecord.model_validate(user) if user else None

        return self._run(inner)

    def get_user_by_email(self, email: str) -> UserRecord | None:
        def inner():
            with self._session() as session:
                user = session.scalar(select(User).where(User.email == email))
                return UserRecord.model_validate(user) if user else None

        return self._run(inner)

    def create_organization(self, slug: str, name: str) -> OrganizationRecord:
        def inner():
            with self._session() as session:
                org = Organization(slug=slug, name=name)
                session.add(org)
                session.commit()
                session.refresh(org)
                return OrganizationRecord.model_validate(org)

        return self._run(inner)

    def create_workspace(self, organization_id: str, slug: str, name: str) -> WorkspaceRecord:
        def inner():
            with self._session() as session:
                workspace = Workspace(
                    organization_id=organization_id, slug=slug, name=name, index_version=1
                )
                session.add(workspace)
                session.commit()
                session.refresh(workspace)
                return WorkspaceRecord.model_validate(workspace)

        return self._run(inner)

    def add_membership(
        self, user_id: str, organization_id: str, workspace_id: str, role: MembershipRole
    ) -> MembershipRecord:
        def inner():
            with self._session() as session:
                membership = Membership(
                    user_id=user_id,
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                    role=role.value,
                )
                session.add(membership)
                session.commit()
                session.refresh(membership)
                return MembershipRecord.model_validate(
                    {**membership.__dict__, "role": membership.role}
                )

        return self._run(inner)

    def get_membership(self, user_id: str, workspace_id: str) -> MembershipRecord | None:
        def inner():
            with self._session() as session:
                membership = session.scalar(
                    select(Membership).where(
                        Membership.user_id == user_id,
                        Membership.workspace_id == workspace_id,
                    )
                )
                return (
                    MembershipRecord.model_validate(
                        {**membership.__dict__, "role": membership.role}
                    )
                    if membership
                    else None
                )

        return self._run(inner)

    def list_user_workspaces(self, user_id: str) -> list[WorkspaceMembership]:
        def inner():
            with self._session() as session:
                rows = session.execute(
                    select(Membership, Workspace, Organization)
                    .join(Workspace, Workspace.id == Membership.workspace_id)
                    .join(Organization, Organization.id == Membership.organization_id)
                    .where(Membership.user_id == user_id)
                ).all()
                return [
                    WorkspaceMembership(
                        workspace=WorkspaceRecord.model_validate(workspace),
                        organization=OrganizationRecord.model_validate(organization),
                        role=MembershipRole(membership.role),
                    )
                    for membership, workspace, organization in rows
                ]

        return self._run(inner)

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        def inner():
            with self._session() as session:
                workspace = session.get(Workspace, workspace_id)
                return WorkspaceRecord.model_validate(workspace) if workspace else None

        return self._run(inner)

    def get_document(self, document_id: str, workspace_id: str) -> DocumentRecord | None:
        def inner():
            with self._session() as session:
                document = session.scalar(
                    select(Document).where(
                        Document.id == document_id, Document.workspace_id == workspace_id
                    )
                )
                return self._document_record(document) if document else None

        return self._run(inner)

    def list_documents(
        self, workspace_id: str, *, limit: int, after: DocumentCursor | None = None
    ) -> list[DocumentRecord]:
        def inner():
            query = select(Document).where(Document.workspace_id == workspace_id)
            if after is not None:
                # Row comparison matches ix_documents_workspace_created_id, so each page
                # is an index range scan however deep the client has paged.
                query = query.where(
                    tuple_(Document.created_at, Document.id)
                    > tuple_(
                        literal(after.created_at, Document.created_at.type),
                        literal(after.id, Document.id.type),
                    )
                )
            query = query.order_by(Document.created_at, Document.id).limit(limit)
            with self._session() as session:
                return [self._document_record(doc) for doc in session.scalars(query).all()]

        return self._run(inner)

    def find_active_document_by_sha(self, workspace_id: str, sha256: str) -> DocumentRecord | None:
        def inner():
            with self._session() as session:
                document = session.scalar(
                    select(Document).where(
                        Document.workspace_id == workspace_id,
                        Document.sha256 == sha256,
                        _ACTIVE_DOCUMENT_PREDICATE,
                    )
                )
                return self._document_record(document) if document else None

        return self._run(inner)

    def publish_document(
        self,
        document: DocumentRecord,
        chunks: list[ChunkRecord],
        job: IngestionJobRecord,
        embedding_model: str,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        def inner():
            try:
                with self._session() as session, session.begin():
                    row = self._document_model(document)
                    session.add(row)
                    # Flush first so a concurrent duplicate fails here, before the chunks
                    # are written; the partial unique index is the arbiter.
                    session.flush()
                    session.add_all(self._chunk_model(chunk, embedding_model) for chunk in chunks)
                    job_row = self._job_model(job)
                    session.add(job_row)
                    self._bump_index_version(session, document.workspace_id, len(chunks))
                    session.flush()
                    session.refresh(row)
                    return self._document_record(row), self._job_record(job_row)
            except IntegrityError as exc:
                if _constraint_name(exc) != _ACTIVE_SHA_INDEX:
                    raise
            existing = self.find_active_document_by_sha(document.workspace_id, document.sha256)
            raise DuplicateDocumentConflict(existing.id if existing else "")

        return self._run(inner)

    def record_failed_document(
        self, document: DocumentRecord, job: IngestionJobRecord
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        def inner():
            with self._session() as session, session.begin():
                row = self._document_model(document)
                job_row = self._job_model(job)
                session.add_all([row, job_row])
                session.flush()
                session.refresh(row)
                return self._document_record(row), self._job_record(job_row)

        return self._run(inner)

    def replace_document_index(
        self,
        document_id: str,
        workspace_id: str,
        chunks: list[ChunkRecord],
        job: IngestionJobRecord,
        embedding_model: str,
        metadata: dict,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        def inner():
            with self._session() as session, session.begin():
                row = self._lock_document(session, document_id, workspace_id)
                if row is None or row.status == DocumentStatus.deleted.value:
                    raise DocumentNotFoundError(document_id)
                removed = session.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
                ).rowcount
                session.add_all(self._chunk_model(chunk, embedding_model) for chunk in chunks)
                row.status = DocumentStatus.ready.value
                row.document_metadata = metadata
                row.error_code = None
                row.error_message = None
                row.updated_at = datetime.now(UTC)
                job_row = self._job_model(job)
                session.add(job_row)
                self._bump_index_version(session, workspace_id, len(chunks) - removed)
                session.flush()
                return self._document_record(row), self._job_record(job_row)

        return self._run(inner)

    def delete_document(
        self, document_id: str, workspace_id: str, job: IngestionJobRecord
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        def inner():
            with self._session() as session, session.begin():
                row = self._lock_document(session, document_id, workspace_id)
                if row is None:
                    raise DocumentNotFoundError(document_id)
                removed = session.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
                ).rowcount
                row.status = DocumentStatus.deleted.value
                row.updated_at = datetime.now(UTC)
                job_row = self._job_model(job)
                session.add(job_row)
                self._bump_index_version(session, workspace_id, -removed)
                session.flush()
                return self._document_record(row), self._job_record(job_row)

        return self._run(inner)

    @staticmethod
    def _lock_document(session: Session, document_id: str, workspace_id: str) -> Document | None:
        """Row-lock the document so reindex and delete of the same document serialize."""
        return session.scalar(
            select(Document)
            .where(Document.id == document_id, Document.workspace_id == workspace_id)
            .with_for_update()
        )

    @staticmethod
    def _bump_index_version(session: Session, workspace_id: str, chunk_delta: int) -> None:
        """New index version (invalidates caches) and adjusted chunk count, in the caller's
        transaction, so the count can never drift from the chunks actually stored."""
        session.execute(
            update(Workspace)
            .where(Workspace.id == workspace_id)
            .values(
                index_version=Workspace.index_version + 1,
                chunk_count=Workspace.chunk_count + chunk_delta,
                updated_at=datetime.now(UTC),
            )
        )

    @staticmethod
    def _document_model(record: DocumentRecord) -> Document:
        return Document(
            id=record.id,
            workspace_id=record.workspace_id,
            created_by_user_id=record.created_by_user_id,
            source_name=record.source_name,
            media_type=record.media_type,
            file_size=record.file_size,
            sha256=record.sha256,
            status=record.status.value,
            content_text=record.content_text,
            document_metadata=record.metadata,
            error_code=record.error_code,
            error_message=record.error_message,
        )

    @staticmethod
    def _chunk_model(chunk: ChunkRecord, embedding_model: str) -> DocumentChunk:
        return DocumentChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            workspace_id=chunk.workspace_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            token_count=chunk.token_count,
            embedding=chunk.embedding,
            embedding_model=embedding_model,
            chunk_metadata=chunk.metadata,
        )

    @staticmethod
    def _job_model(record: IngestionJobRecord) -> IngestionJob:
        return IngestionJob(
            id=record.id,
            workspace_id=record.workspace_id,
            requested_by_user_id=record.requested_by_user_id,
            document_id=record.document_id,
            operation=record.operation,
            status=record.status.value,
            attempt_count=record.attempt_count,
            error_code=record.error_code,
            error_message=record.error_message,
            completed_at=record.completed_at,
        )

    def dense_search(
        self, workspace_id: str, query_embedding: list[float], limit: int
    ) -> tuple[list[RetrievedChunk], float]:
        """Nearest chunks within one workspace, by one of two strategies.

        The HNSW index spans every workspace and the tenant filter applies to what it returns.
        By default pgvector stops after ``hnsw.ef_search`` candidates (40), so a workspace
        holding a small share of the table can get fewer results than requested, or none.

        * Workspaces up to ``exact_search_max_chunks`` get an exact search over their own
          rows (workspace index, then sort): complete, and cheap at that size.
        * Larger workspaces use HNSW with ``hnsw.iterative_scan`` (pgvector 0.8+), which
          keeps scanning until ``limit`` rows pass the filter. ``relaxed_order`` may return
          them slightly out of order, so the outer query re-sorts by distance.
        """
        vector_literal = "[" + ",".join(f"{value:.8f}" for value in query_embedding) + "]"

        def inner():
            start = time.perf_counter()
            with self._session() as session, session.begin():
                chunk_count = session.scalar(
                    select(Workspace.chunk_count).where(Workspace.id == workspace_id)
                )
                exact = (chunk_count or 0) <= self._exact_search_max_chunks
                if not exact and self._supports_iterative_scan(session):
                    session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
                rows = (
                    session.execute(
                        _EXACT_DENSE_SQL if exact else _ANN_DENSE_SQL,
                        {"workspace_id": workspace_id, "embedding": vector_literal, "limit": limit},
                    )
                    .mappings()
                    .all()
                )
            latency_ms = (time.perf_counter() - start) * 1000
            return [
                RetrievedChunk(
                    chunk_id=str(row["id"]),
                    document_id=str(row["document_id"]),
                    source_name=row["source_name"],
                    content=row["content"],
                    fused_score=float(row["score"] or 0.0),
                    vector_score=float(row["score"] or 0.0),
                    metadata=row["chunk_metadata"] or {},
                )
                for row in rows
            ], latency_ms

        return self._run(inner)

    def _supports_iterative_scan(self, session: Session) -> bool:
        """pgvector 0.8 added hnsw.iterative_scan; older versions reject the setting."""
        if self._pgvector_version is None:
            version = session.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            self._pgvector_version = tuple(
                int(part) for part in str(version or "0").split(".")[:3] if part.isdigit()
            )
        return self._pgvector_version >= (0, 8, 0)

    def lexical_search(
        self, workspace_id: str, query: str, limit: int
    ) -> tuple[list[RetrievedChunk], float]:
        sql = text(
            """
            SELECT c.id, c.document_id, d.source_name, c.content, c.chunk_metadata,
                   ts_rank_cd(
                       to_tsvector('english', c.content),
                       websearch_to_tsquery('english', :query)
                   ) AS score
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.workspace_id = :workspace_id
              AND d.status = 'ready'
              AND to_tsvector('english', c.content) @@ websearch_to_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :limit
            """
        )

        def inner():
            start = time.perf_counter()
            with self._session() as session:
                rows = (
                    session.execute(
                        sql, {"workspace_id": workspace_id, "query": query, "limit": limit}
                    )
                    .mappings()
                    .all()
                )
            latency_ms = (time.perf_counter() - start) * 1000
            return [
                RetrievedChunk(
                    chunk_id=str(row["id"]),
                    document_id=str(row["document_id"]),
                    source_name=row["source_name"],
                    content=row["content"],
                    fused_score=float(row["score"] or 0.0),
                    lexical_score=float(row["score"] or 0.0),
                    metadata=row["chunk_metadata"] or {},
                )
                for row in rows
            ], latency_ms

        return self._run(inner)

    def create_ingestion_job(self, record: IngestionJobRecord) -> IngestionJobRecord:
        def inner():
            with self._session() as session, session.begin():
                job = self._job_model(record)
                session.add(job)
                session.flush()
                return self._job_record(job)

        return self._run(inner)

    def get_ingestion_job(self, job_id: str) -> IngestionJobRecord | None:
        def inner():
            with self._session() as session:
                job = session.get(IngestionJob, job_id)
                return self._job_record(job) if job else None

        return self._run(inner)

    def term_document_frequencies(
        self, workspace_id: str, stems: list[str]
    ) -> tuple[dict[str, int], int]:
        """Chunk frequency of each stem, matching the in-memory store's ``stem_matches``.

        The full-text GIN index (``ix_document_chunks_fts``) first narrows each count to
        chunks with a lexeme starting with the stem (``stem:*``); the exact word-boundary
        regex then runs only on those candidates. A word whose English lexeme is shorter
        than the stem ("running" -> "run" for stem "runn") is missed, which only lowers
        that term's frequency. Stems are lowercase ``[a-z0-9]+``, safe in both tsquery and
        regex syntax. Chunks exist only for ready documents, so no join is needed.
        """
        sql = text(
            r"""
            SELECT t.stem,
                   (SELECT count(*)
                      FROM document_chunks c
                     WHERE c.workspace_id = :workspace_id
                       AND to_tsvector('english', c.content)
                           @@ to_tsquery('english', t.stem || ':*')
                       AND c.content ~* CASE
                             WHEN length(t.stem) < 3 THEN '\m' || t.stem || '\M'
                             ELSE '\m' || t.stem || '[a-z0-9]{0,' || :extra || '}\M'
                           END) AS df
            FROM unnest(CAST(:stems AS text[])) AS t(stem)
            """
        )

        def inner():
            if not stems:
                return {}, 0
            params = {"workspace_id": workspace_id, "stems": list(stems), "extra": STEM_MAX_EXTRA}
            with self._session() as session:
                rows = session.execute(sql, params).all()
                total = int(
                    session.scalar(
                        select(Workspace.chunk_count).where(Workspace.id == workspace_id)
                    )
                    or 0
                )
            return {row.stem: int(row.df) for row in rows}, total

        return self._run(inner)

    def corpus_counts(self) -> tuple[int, int]:
        def inner():
            with self._session() as session:
                docs = int(
                    session.scalar(
                        select(func.count()).select_from(Document).where(Document.status == "ready")
                    )
                    or 0
                )
                chunks = int(session.scalar(select(func.count()).select_from(DocumentChunk)) or 0)
                return docs, chunks

        return self._run(inner)

    @staticmethod
    def _document_record(document: Document) -> DocumentRecord:
        return DocumentRecord.model_validate(
            {
                "id": document.id,
                "workspace_id": document.workspace_id,
                "created_by_user_id": document.created_by_user_id,
                "source_name": document.source_name,
                "media_type": document.media_type,
                "file_size": document.file_size,
                "sha256": document.sha256,
                "status": document.status,
                "content_text": document.content_text,
                "metadata": document.document_metadata or {},
                "error_code": document.error_code,
                "error_message": document.error_message,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
            }
        )

    @staticmethod
    def _job_record(job: IngestionJob) -> IngestionJobRecord:
        return IngestionJobRecord.model_validate(
            {
                "id": job.id,
                "workspace_id": job.workspace_id,
                "requested_by_user_id": job.requested_by_user_id,
                "document_id": job.document_id,
                "operation": job.operation,
                "status": job.status,
                "attempt_count": job.attempt_count,
                "error_code": job.error_code,
                "error_message": job.error_message,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "completed_at": job.completed_at,
            }
        )
