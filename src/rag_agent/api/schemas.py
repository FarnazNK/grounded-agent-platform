from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag_agent.schemas import (
    DocumentStatus,
    IngestionJobRecord,
    QueryResult,
    WorkspaceMembership,
)


class BootstrapRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8, max_length=256)
    organization_slug: str = Field(..., min_length=2, max_length=100, pattern=r"^[a-z0-9-]+$")
    organization_name: str = Field(..., min_length=1, max_length=255)
    workspace_slug: str = Field(..., min_length=2, max_length=100, pattern=r"^[a-z0-9-]+$")
    workspace_name: str = Field(..., min_length=1, max_length=255)


class LoginRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QueryRequest(BaseModel):
    workspace_id: str
    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    result: QueryResult


class DocumentView(BaseModel):
    """Public document metadata. Raw document text is deliberately not returned."""

    id: str
    workspace_id: str
    created_by_user_id: str
    source_name: str
    media_type: str
    file_size: int
    sha256: str
    status: DocumentStatus
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    document: DocumentView
    job: IngestionJobRecord | None = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentView]


class IngestionJobResponse(BaseModel):
    job: IngestionJobRecord


class MeResponse(BaseModel):
    user_id: str
    email: str


class WorkspaceListResponse(BaseModel):
    workspaces: list[WorkspaceMembership]


class ErrorResponse(BaseModel):
    error: str
    message: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ExtractionRequest(BaseModel):
    workspace_id: str
    schema_name: Literal["policy_facts"] = Field("policy_facts", alias="schema")

    model_config = {"populate_by_name": True}


class ExtractionResponse(BaseModel):
    document_id: str
    schema_name: str = Field(serialization_alias="schema")
    status: Literal["valid"]
    data: dict[str, Any]
    provider: str
    attempts: int
    latency_ms: float
