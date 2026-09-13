from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MembershipRole(StrEnum):
    viewer = 'viewer'
    editor = 'editor'
    admin = 'admin'


class DocumentStatus(StrEnum):
    pending = 'pending'
    processing = 'processing'
    ready = 'ready'
    failed = 'failed'
    deleted = 'deleted'


class IngestionStatus(StrEnum):
    pending = 'pending'
    processing = 'processing'
    completed = 'completed'
    failed = 'failed'
    duplicate = 'duplicate'


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    content: str
    fused_score: float
    vector_score: float | None = None
    lexical_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_citation(self) -> str:
        return f'[source: {self.source_name}]'


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class LatencyBreakdown(BaseModel):
    total_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    db_ms: float = 0.0


class QueryResult(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    answer: str
    citations: list[str] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    cached: bool = False
    grounded: bool = True
    sanitized_query: str | None = None
    refusal: bool = False


class OrganizationRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    slug: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {'from_attributes': True}


class WorkspaceRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    organization_id: str
    slug: str
    name: str
    index_version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {'from_attributes': True}


class UserRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    password_hash: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {'from_attributes': True}


class MembershipRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    organization_id: str
    workspace_id: str
    role: MembershipRole = MembershipRole.viewer
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {'from_attributes': True}


class WorkspaceMembership(BaseModel):
    workspace: WorkspaceRecord
    organization: OrganizationRecord
    role: MembershipRole


class DocumentRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str
    created_by_user_id: str
    source_name: str
    media_type: str
    file_size: int
    sha256: str
    status: DocumentStatus = DocumentStatus.pending
    content_text: str = ''
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = {'from_attributes': True}


class ChunkRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    workspace_id: str
    chunk_index: int
    content: str
    token_count: int
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionJobRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str
    requested_by_user_id: str
    document_id: str | None = None
    operation: str = 'upload'
    status: IngestionStatus = IngestionStatus.pending
    attempt_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    model_config = {'from_attributes': True}
