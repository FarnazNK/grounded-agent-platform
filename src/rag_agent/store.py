from __future__ import annotations

import math
import re
import threading
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from rag_agent.errors import DatabaseUnavailableError, DocumentNotFoundError
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

# Documents that count for duplicate detection. Failed uploads are excluded so the same
# file can be retried; deleted documents are excluded so it can be re-added.
ACTIVE_DOCUMENT_STATUSES = frozenset(
    {DocumentStatus.pending, DocumentStatus.processing, DocumentStatus.ready}
)


@dataclass(frozen=True)
class DocumentCursor:
    """Position after the last document of a page: its (created_at, id) sort key."""

    created_at: datetime
    id: str


class DuplicateDocumentConflict(Exception):
    """Raised by ``publish_document`` when an active document has the same content."""

    def __init__(self, existing_document_id: str) -> None:
        super().__init__(existing_document_id)
        self.existing_document_id = existing_document_id


class RAGStore(Protocol):
    def ping(self) -> None: ...
    def user_count(self) -> int: ...
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
        """Atomically create the first admin; return None if any user already exists."""
        ...

    def create_user(self, email: str, password_hash: str) -> UserRecord: ...
    def get_user(self, user_id: str) -> UserRecord | None: ...
    def get_user_by_email(self, email: str) -> UserRecord | None: ...
    def create_organization(self, slug: str, name: str) -> OrganizationRecord: ...
    def create_workspace(self, organization_id: str, slug: str, name: str) -> WorkspaceRecord: ...
    def add_membership(
        self, user_id: str, organization_id: str, workspace_id: str, role: MembershipRole
    ) -> MembershipRecord: ...
    def get_membership(self, user_id: str, workspace_id: str) -> MembershipRecord | None: ...
    def list_user_workspaces(self, user_id: str) -> list[WorkspaceMembership]: ...
    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None: ...
    def get_document(self, document_id: str, workspace_id: str) -> DocumentRecord | None: ...
    def list_documents(
        self, workspace_id: str, *, limit: int, after: DocumentCursor | None = None
    ) -> list[DocumentRecord]:
        """Documents ordered by (created_at, id), starting after ``after`` (keyset paging)."""
        ...

    def find_active_document_by_sha(
        self, workspace_id: str, sha256: str
    ) -> DocumentRecord | None: ...
    def publish_document(
        self,
        document: DocumentRecord,
        chunks: list[ChunkRecord],
        job: IngestionJobRecord,
        embedding_model: str,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        """In one transaction: insert a ready document, its chunks and a completed job, and
        bump the workspace index version. Raises :class:`DuplicateDocumentConflict` if an
        active document with the same content already exists in the workspace."""
        ...

    def record_failed_document(
        self, document: DocumentRecord, job: IngestionJobRecord
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        """In one transaction: insert a failed document and its failed job."""
        ...

    def replace_document_index(
        self,
        document_id: str,
        workspace_id: str,
        chunks: list[ChunkRecord],
        job: IngestionJobRecord,
        embedding_model: str,
        metadata: dict,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        """In one transaction, with the document row locked: swap in new chunks, mark the
        document ready, record the job and bump the index version. Raises
        :class:`DocumentNotFoundError` if the document is missing or deleted."""
        ...

    def delete_document(
        self, document_id: str, workspace_id: str, job: IngestionJobRecord
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        """In one transaction, with the document row locked: remove its chunks, mark it
        deleted, record the job and bump the index version."""
        ...

    def dense_search(
        self, workspace_id: str, query_embedding: list[float], limit: int
    ) -> tuple[list[RetrievedChunk], float]: ...
    def lexical_search(
        self, workspace_id: str, query: str, limit: int
    ) -> tuple[list[RetrievedChunk], float]: ...
    def create_ingestion_job(self, record: IngestionJobRecord) -> IngestionJobRecord: ...
    def get_ingestion_job(self, job_id: str) -> IngestionJobRecord | None: ...
    def corpus_counts(self) -> tuple[int, int]: ...
    def term_document_frequencies(
        self, workspace_id: str, stems: list[str]
    ) -> tuple[dict[str, int], int]:
        """Number of ready chunks with a word matching each stem, and the total.

        A word matches a stem when it starts with it and is at most ``STEM_MAX_EXTRA``
        characters longer (see :func:`stem_matches`).
        """
        ...


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
        self._bootstrap_lock = threading.Lock()
        # Serializes multi-step writes, standing in for a database transaction.
        self._write_lock = threading.RLock()

    def ping(self) -> None:
        if not self.available:
            raise DatabaseUnavailableError()

    def user_count(self) -> int:
        return len(self.users)

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
        with self._bootstrap_lock:
            if self.users:
                return None
            user = self.create_user(email, password_hash)
            org = self.create_organization(organization_slug, organization_name)
            workspace = self.create_workspace(org.id, workspace_slug, workspace_name)
            self.add_membership(user.id, org.id, workspace.id, MembershipRole.admin)
            return WorkspaceMembership(
                workspace=workspace, organization=org, role=MembershipRole.admin
            )

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

    def add_membership(
        self, user_id: str, organization_id: str, workspace_id: str, role: MembershipRole
    ) -> MembershipRecord:
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
            results.append(
                WorkspaceMembership(workspace=workspace, organization=org, role=membership.role)
            )
        return results

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        return deepcopy(self.workspaces.get(workspace_id))

    def get_document(self, document_id: str, workspace_id: str) -> DocumentRecord | None:
        record = self.documents.get(document_id)
        if record and record.workspace_id == workspace_id:
            return deepcopy(record)
        return None

    def list_documents(
        self, workspace_id: str, *, limit: int, after: DocumentCursor | None = None
    ) -> list[DocumentRecord]:
        documents = sorted(
            (doc for doc in self.documents.values() if doc.workspace_id == workspace_id),
            key=lambda doc: (doc.created_at, doc.id),
        )
        if after is not None:
            documents = [
                doc for doc in documents if (doc.created_at, doc.id) > (after.created_at, after.id)
            ]
        return [deepcopy(doc) for doc in documents[:limit]]

    def find_active_document_by_sha(self, workspace_id: str, sha256: str) -> DocumentRecord | None:
        for record in self.documents.values():
            if (
                record.workspace_id == workspace_id
                and record.sha256 == sha256
                and record.status in ACTIVE_DOCUMENT_STATUSES
            ):
                return deepcopy(record)
        return None

    def publish_document(
        self,
        document: DocumentRecord,
        chunks: list[ChunkRecord],
        job: IngestionJobRecord,
        embedding_model: str,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        with self._write_lock:
            existing = self.find_active_document_by_sha(document.workspace_id, document.sha256)
            if existing is not None:
                raise DuplicateDocumentConflict(existing.id)
            self.documents[document.id] = deepcopy(document)
            for chunk in chunks:
                self.chunks[chunk.id] = deepcopy(chunk)
            self.ingestion_jobs[job.id] = deepcopy(job)
            self._bump_index_version(document.workspace_id)
            return deepcopy(document), deepcopy(job)

    def record_failed_document(
        self, document: DocumentRecord, job: IngestionJobRecord
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        with self._write_lock:
            self.documents[document.id] = deepcopy(document)
            self.ingestion_jobs[job.id] = deepcopy(job)
            return deepcopy(document), deepcopy(job)

    def replace_document_index(
        self,
        document_id: str,
        workspace_id: str,
        chunks: list[ChunkRecord],
        job: IngestionJobRecord,
        embedding_model: str,
        metadata: dict,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        with self._write_lock:
            current = self.documents.get(document_id)
            if (
                current is None
                or current.workspace_id != workspace_id
                or current.status == DocumentStatus.deleted
            ):
                raise DocumentNotFoundError(document_id)
            self._delete_chunks(document_id)
            for chunk in chunks:
                self.chunks[chunk.id] = deepcopy(chunk)
            updated = current.model_copy(
                update={
                    "status": DocumentStatus.ready,
                    "metadata": metadata,
                    "error_code": None,
                    "error_message": None,
                    "updated_at": datetime.now(UTC),
                }
            )
            self.documents[document_id] = updated
            self.ingestion_jobs[job.id] = deepcopy(job)
            self._bump_index_version(workspace_id)
            return deepcopy(updated), deepcopy(job)

    def delete_document(
        self, document_id: str, workspace_id: str, job: IngestionJobRecord
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        with self._write_lock:
            current = self.documents.get(document_id)
            if current is None or current.workspace_id != workspace_id:
                raise DocumentNotFoundError(document_id)
            self._delete_chunks(document_id)
            updated = current.model_copy(
                update={"status": DocumentStatus.deleted, "updated_at": datetime.now(UTC)}
            )
            self.documents[document_id] = updated
            self.ingestion_jobs[job.id] = deepcopy(job)
            self._bump_index_version(workspace_id)
            return deepcopy(updated), deepcopy(job)

    def _delete_chunks(self, document_id: str) -> None:
        for chunk_id in [
            key for key, value in self.chunks.items() if value.document_id == document_id
        ]:
            self.chunks.pop(chunk_id, None)

    def _bump_index_version(self, workspace_id: str) -> None:
        workspace = self.workspaces[workspace_id]
        chunk_count = sum(1 for chunk in self.chunks.values() if chunk.workspace_id == workspace_id)
        self.workspaces[workspace_id] = workspace.model_copy(
            update={"index_version": workspace.index_version + 1, "chunk_count": chunk_count}
        )

    def term_document_frequencies(
        self, workspace_id: str, stems: list[str]
    ) -> tuple[dict[str, int], int]:
        chunk_terms: list[set[str]] = []
        for chunk in self.chunks.values():
            if chunk.workspace_id != workspace_id:
                continue
            doc = self.documents.get(chunk.document_id)
            if doc and doc.status == DocumentStatus.ready:
                chunk_terms.append(set(_terms(chunk.content)))
        return (
            {
                value: sum(
                    any(stem_matches(value, word) for word in present) for present in chunk_terms
                )
                for value in stems
            },
            len(chunk_terms),
        )

    def dense_search(
        self, workspace_id: str, query_embedding: list[float], limit: int
    ) -> tuple[list[RetrievedChunk], float]:
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

    def lexical_search(
        self, workspace_id: str, query: str, limit: int
    ) -> tuple[list[RetrievedChunk], float]:
        query_terms = _terms(query)
        if not query_terms:
            return [], 0.0

        corpus: list[tuple[ChunkRecord, DocumentRecord, list[str]]] = []
        for chunk in self.chunks.values():
            if chunk.workspace_id != workspace_id:
                continue
            doc = self.documents.get(chunk.document_id)
            if not doc or doc.status != DocumentStatus.ready:
                continue
            corpus.append((chunk, doc, _terms(chunk.content)))
        if not corpus:
            return [], 0.0

        document_frequency = {
            term: sum(1 for _, _, terms in corpus if term in set(terms))
            for term in set(query_terms)
        }
        average_length = sum(len(terms) for _, _, terms in corpus) / len(corpus)
        k1 = 1.2
        b = 0.75
        scored: list[tuple[float, ChunkRecord, DocumentRecord]] = []
        for chunk, doc, terms in corpus:
            counts = Counter(terms)
            score = 0.0
            for term in query_terms:
                tf = counts[term]
                if not tf:
                    continue
                df = document_frequency[term]
                idf = math.log(1.0 + (len(corpus) - df + 0.5) / (df + 0.5))
                norm = tf + k1 * (1.0 - b + b * len(terms) / max(1.0, average_length))
                score += idf * (tf * (k1 + 1.0) / norm)
            if score > 0:
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

    def get_ingestion_job(self, job_id: str) -> IngestionJobRecord | None:
        record = self.ingestion_jobs.get(job_id)
        return deepcopy(record) if record else None

    def corpus_counts(self) -> tuple[int, int]:
        document_count = sum(
            1 for doc in self.documents.values() if doc.status == DocumentStatus.ready
        )
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


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


_STEM_SUFFIXES = ("ies", "ing", "ed", "es", "s")
# How many extra characters a word may have beyond a stem and still count as a match.
STEM_MAX_EXTRA = 4


def stem(term: str) -> str:
    """Strip one common English suffix: "backs"/"backed" -> "back", "policies" -> "polic"."""
    for suffix in _STEM_SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            return term[: -len(suffix)]
    return term


def stem_matches(stem_value: str, word: str) -> bool:
    """True if ``word`` is ``stem_value`` plus at most a short suffix ("back" ~ "backed")."""
    if len(stem_value) < 3:
        return word == stem_value
    return word.startswith(stem_value) and len(word) - len(stem_value) <= STEM_MAX_EXTRA


def _terms(text: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in _STOP_WORDS and len(term) > 1
    ]
