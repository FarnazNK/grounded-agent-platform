from __future__ import annotations

import math
from copy import deepcopy
from datetime import UTC, datetime
from typing import Protocol

from rag_agent.errors import DatabaseUnavailableError
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


class RAGStore(Protocol):
    def ping(self) -> None: ...
    def user_count(self) -> int: ...
    def create_user(self, email: str, password_hash: str) -> UserRecord: ...
    def get_user(self, user_id: str) -> UserRecord | None: ...
    def get_user_by_email(self, email: str) -> UserRecord | None: ...
    def create_organization(self, slug: str, name: str) -> OrganizationRecord: ...
    def create_workspace(self, organization_id: str, slug: str, name: str) -> WorkspaceRecord: ...
    def add_membership(self, user_id: str, organization_id: str, workspace_id: str, role: MembershipRole) -> MembershipRecord: ...
    def get_membership(self, user_id: str, workspace_id: str) -> MembershipRecord | None: ...
    def list_user_workspaces(self, user_id: str) -> list[WorkspaceMembership]: ...
    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None: ...
    def create_document(self, record: DocumentRecord) -> DocumentRecord: ...
    def update_document(self, document_id: str, **updates) -> DocumentRecord: ...
    def get_document(self, document_id: str, workspace_id: str) -> DocumentRecord | None: ...
    def list_documents(self, workspace_id: str) -> list[DocumentRecord]: ...
    def find_active_document_by_sha(self, workspace_id: str, sha256: str) -> DocumentRecord | None: ...
    def replace_document_chunks(self, document_id: str, workspace_id: str, chunks: list[ChunkRecord], embedding_model: str) -> None: ...
    def delete_document_chunks(self, document_id: str) -> None: ...
    def dense_search(self, workspace_id: str, query_embedding: list[float], limit: int) -> tuple[list[RetrievedChunk], float]: ...
    def lexical_search(self, workspace_id: str, query: str, limit: int) -> tuple[list[RetrievedChunk], float]: ...
    def create_ingestion_job(self, record: IngestionJobRecord) -> IngestionJobRecord: ...
    def update_ingestion_job(self, job_id: str, **updates) -> IngestionJobRecord: ...
    def get_ingestion_job(self, job_id: str) -> IngestionJobRecord | None: ...
    def bump_workspace_index_version(self, workspace_id: str) -> WorkspaceRecord: ...
    def corpus_counts(self) -> tuple[int, int]: ...


class InMemoryRAGStore:
    def __init__(self) -> None:
        self.users: dict[str, UserRecord] = {}
        self.organizations: dict[str, OrganizationRecord] = {}
        self.workspaces: dict[str, WorkspaceRecord] = {}
        self.memberships: dict[str, MembershipRecord] = {}
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self.ingestion_jobs: dict[str, IngestionJobRecord] = {}
        self.available = True

    def ping(self) -> None:
        if not self.available:
            raise DatabaseUnavailableError()

    def user_count(self) -> int:
        return len(self.users)

    def create_user(self, email: str, password_hash: str) -> UserRecord:
        record = UserRecord(email=email, password_hash=password_hash)
        self.users[record.id] = record
        return record

    def get_user(self, user_id: str) -> UserRecord | None:
        return deepcopy(self.users.get(user_id))

    def get_user_by_email(self, email: str) -> UserRecord | None:
        for user in self.users.values():
            if user.email == email:
                return deepcopy(user)
        return None

    def create_organization(self, slug: str, name: str) -> OrganizationRecord:
        record = OrganizationRecord(slug=slug, name=name)
        self.organizations[record.id] = record
        return record

    def create_workspace(self, organization_id: str, slug: str, name: str) -> WorkspaceRecord:
        record = WorkspaceRecord(organization_id=organization_id, slug=slug, name=name)
        self.workspaces[record.id] = record
        return record

    def add_membership(self, user_id: str, organization_id: str, workspace_id: str, role: MembershipRole) -> MembershipRecord:
        record = MembershipRecord(
            user_id=user_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            role=role,
        )
        self.memberships[record.id] = record
        return record

    def get_membership(self, user_id: str, workspace_id: str) -> MembershipRecord | None:
        for record in self.memberships.values():
            if record.user_id == user_id and record.workspace_id == workspace_id:
                return deepcopy(record)
        return None

    def list_user_workspaces(self, user_id: str) -> list[WorkspaceMembership]:
        results: list[WorkspaceMembership] = []
        for membership in self.memberships.values():
            if membership.user_id != user_id:
                continue
            workspace = self.workspaces[membership.workspace_id]
            org = self.organizations[membership.organization_id]
            results.append(WorkspaceMembership(workspace=workspace, organization=org, role=membership.role))
        return results

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        return deepcopy(self.workspaces.get(workspace_id))

    def create_document(self, record: DocumentRecord) -> DocumentRecord:
        self.documents[record.id] = deepcopy(record)
        return deepcopy(record)

    def update_document(self, document_id: str, **updates) -> DocumentRecord:
        record = self.documents[document_id].model_copy(deep=True)
        data = record.model_dump()
        data.update(updates)
        data['updated_at'] = datetime.now(UTC)
        updated = DocumentRecord.model_validate(data)
        self.documents[document_id] = updated
        return deepcopy(updated)

    def get_document(self, document_id: str, workspace_id: str) -> DocumentRecord | None:
        record = self.documents.get(document_id)
        if record and record.workspace_id == workspace_id:
            return deepcopy(record)
        return None

    def list_documents(self, workspace_id: str) -> list[DocumentRecord]:
        return [deepcopy(doc) for doc in self.documents.values() if doc.workspace_id == workspace_id]

    def find_active_document_by_sha(self, workspace_id: str, sha256: str) -> DocumentRecord | None:
        for record in self.documents.values():
            if record.workspace_id == workspace_id and record.sha256 == sha256 and record.status != DocumentStatus.deleted:
                return deepcopy(record)
        return None

    def replace_document_chunks(self, document_id: str, workspace_id: str, chunks: list[ChunkRecord], embedding_model: str) -> None:
        self.delete_document_chunks(document_id)
        for chunk in chunks:
            self.chunks[chunk.id] = deepcopy(chunk)

    def delete_document_chunks(self, document_id: str) -> None:
        for chunk_id in [key for key, value in self.chunks.items() if value.document_id == document_id]:
            self.chunks.pop(chunk_id, None)

    def dense_search(self, workspace_id: str, query_embedding: list[float], limit: int) -> tuple[list[RetrievedChunk], float]:
        scored: list[tuple[float, ChunkRecord, DocumentRecord]] = []
        for chunk in self.chunks.values():
            if chunk.workspace_id != workspace_id:
                continue
            doc = self.documents.get(chunk.document_id)
            if not doc or doc.status != DocumentStatus.ready:
                continue
            score = _cosine_similarity(chunk.embedding, query_embedding)
            scored.append((score, chunk, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                source_name=doc.source_name,
                content=chunk.content,
                fused_score=score,
                vector_score=score,
                metadata=deepcopy(chunk.metadata),
            )
            for score, chunk, doc in scored[:limit]
        ], 0.0

    def lexical_search(self, workspace_id: str, query: str, limit: int) -> tuple[list[RetrievedChunk], float]:
        query_terms = {term for term in query.lower().split() if term}
        scored: list[tuple[float, ChunkRecord, DocumentRecord]] = []
        for chunk in self.chunks.values():
            if chunk.workspace_id != workspace_id:
                continue
            doc = self.documents.get(chunk.document_id)
            if not doc or doc.status != DocumentStatus.ready:
                continue
            doc_terms = set(chunk.content.lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap:
                score = overlap / max(1, len(query_terms))
                scored.append((score, chunk, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                source_name=doc.source_name,
                content=chunk.content,
                fused_score=score,
                lexical_score=score,
                metadata=deepcopy(chunk.metadata),
            )
            for score, chunk, doc in scored[:limit]
        ], 0.0

    def create_ingestion_job(self, record: IngestionJobRecord) -> IngestionJobRecord:
        self.ingestion_jobs[record.id] = deepcopy(record)
        return deepcopy(record)

    def update_ingestion_job(self, job_id: str, **updates) -> IngestionJobRecord:
        record = self.ingestion_jobs[job_id].model_copy(deep=True)
        data = record.model_dump()
        data.update(updates)
        data['updated_at'] = datetime.now(UTC)
        updated = IngestionJobRecord.model_validate(data)
        self.ingestion_jobs[job_id] = updated
        return deepcopy(updated)

    def get_ingestion_job(self, job_id: str) -> IngestionJobRecord | None:
        record = self.ingestion_jobs.get(job_id)
        return deepcopy(record) if record else None

    def bump_workspace_index_version(self, workspace_id: str) -> WorkspaceRecord:
        workspace = self.workspaces[workspace_id]
        updated = workspace.model_copy(update={'index_version': workspace.index_version + 1})
        self.workspaces[workspace_id] = updated
        return deepcopy(updated)

    def corpus_counts(self) -> tuple[int, int]:
        document_count = sum(1 for doc in self.documents.values() if doc.status == DocumentStatus.ready)
        return document_count, len(self.chunks)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))
