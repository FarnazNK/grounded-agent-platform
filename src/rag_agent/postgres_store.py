from __future__ import annotations

import time
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from rag_agent.errors import DatabaseUnavailableError
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


class PostgresRAGStore:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def _session(self) -> Session:
        return self._session_factory()

    def _run(self, fn):
        try:
            return fn()
        except OperationalError as exc:
            raise DatabaseUnavailableError() from exc

    def ping(self) -> None:
        self._run(lambda: self._session().execute(text("SELECT 1")))

    def user_count(self) -> int:
        def inner():
            with self._session() as session:
                return int(session.scalar(select(func.count()).select_from(User)) or 0)

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

    def create_document(self, record: DocumentRecord) -> DocumentRecord:
        def inner():
            with self._session() as session:
                document = Document(
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
                session.add(document)
                session.commit()
                session.refresh(document)
                return self._document_record(document)

        return self._run(inner)

    def update_document(self, document_id: str, **updates) -> DocumentRecord:
        def inner():
            with self._session() as session:
                if "status" in updates and hasattr(updates["status"], "value"):
                    updates["status"] = updates["status"].value
                if "metadata" in updates:
                    updates["document_metadata"] = updates.pop("metadata")
                updates["updated_at"] = datetime.now(UTC)
                session.execute(
                    update(Document).where(Document.id == document_id).values(**updates)
                )
                session.commit()
                document = session.get(Document, document_id)
                assert document is not None
                return self._document_record(document)

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

    def list_documents(self, workspace_id: str) -> list[DocumentRecord]:
        def inner():
            with self._session() as session:
                docs = session.scalars(
                    select(Document).where(Document.workspace_id == workspace_id)
                ).all()
                return [self._document_record(doc) for doc in docs]

        return self._run(inner)

    def find_active_document_by_sha(self, workspace_id: str, sha256: str) -> DocumentRecord | None:
        def inner():
            with self._session() as session:
                document = session.scalar(
                    select(Document).where(
                        Document.workspace_id == workspace_id,
                        Document.sha256 == sha256,
                        Document.status != DocumentStatus.deleted.value,
                    )
                )
                return self._document_record(document) if document else None

        return self._run(inner)

    def replace_document_chunks(
        self, document_id: str, workspace_id: str, chunks: list[ChunkRecord], embedding_model: str
    ) -> None:
        def inner():
            with self._session() as session:
                session.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
                )
                for chunk in chunks:
                    session.add(
                        DocumentChunk(
                            id=chunk.id,
                            document_id=document_id,
                            workspace_id=workspace_id,
                            chunk_index=chunk.chunk_index,
                            content=chunk.content,
                            token_count=chunk.token_count,
                            embedding=chunk.embedding,
                            embedding_model=embedding_model,
                            chunk_metadata=chunk.metadata,
                        )
                    )
                session.commit()

        self._run(inner)

    def delete_document_chunks(self, document_id: str) -> None:
        self._run(lambda: self._delete_chunks(document_id))

    def _delete_chunks(self, document_id: str) -> None:
        with self._session() as session:
            session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
            session.commit()

    def dense_search(
        self, workspace_id: str, query_embedding: list[float], limit: int
    ) -> tuple[list[RetrievedChunk], float]:
        vector_literal = "[" + ",".join(f"{value:.8f}" for value in query_embedding) + "]"
        sql = text(
            """
            SELECT c.id, c.document_id, d.source_name, c.content, c.chunk_metadata,
                   1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.workspace_id = :workspace_id AND d.status = 'ready'
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
            """
        )

        def inner():
            start = time.perf_counter()
            with self._session() as session:
                rows = (
                    session.execute(
                        sql,
                        {"workspace_id": workspace_id, "embedding": vector_literal, "limit": limit},
                    )
                    .mappings()
                    .all()
                )
            latency_ms = (time.perf_counter() - start) * 1000
            return [
                RetrievedChunk(
                    chunk_id=row["id"],
                    document_id=row["document_id"],
                    source_name=row["source_name"],
                    content=row["content"],
                    fused_score=float(row["score"] or 0.0),
                    vector_score=float(row["score"] or 0.0),
                    metadata=row["chunk_metadata"] or {},
                )
                for row in rows
            ], latency_ms

        return self._run(inner)

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
                    chunk_id=row["id"],
                    document_id=row["document_id"],
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
            with self._session() as session:
                job = IngestionJob(
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
                session.add(job)
                session.commit()
                session.refresh(job)
                return self._job_record(job)

        return self._run(inner)

    def update_ingestion_job(self, job_id: str, **updates) -> IngestionJobRecord:
        def inner():
            with self._session() as session:
                if "status" in updates and hasattr(updates["status"], "value"):
                    updates["status"] = updates["status"].value
                updates["updated_at"] = datetime.now(UTC)
                session.execute(
                    update(IngestionJob).where(IngestionJob.id == job_id).values(**updates)
                )
                session.commit()
                job = session.get(IngestionJob, job_id)
                assert job is not None
                return self._job_record(job)

        return self._run(inner)

    def get_ingestion_job(self, job_id: str) -> IngestionJobRecord | None:
        def inner():
            with self._session() as session:
                job = session.get(IngestionJob, job_id)
                return self._job_record(job) if job else None

        return self._run(inner)

    def bump_workspace_index_version(self, workspace_id: str) -> WorkspaceRecord:
        def inner():
            with self._session() as session:
                session.execute(
                    update(Workspace)
                    .where(Workspace.id == workspace_id)
                    .values(index_version=Workspace.index_version + 1, updated_at=datetime.now(UTC))
                )
                session.commit()
                workspace = session.get(Workspace, workspace_id)
                assert workspace is not None
                return WorkspaceRecord.model_validate(workspace)

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
