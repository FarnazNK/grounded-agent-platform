from __future__ import annotations

from pydantic import BaseModel, Field

from rag_agent.schemas import DocumentRecord, IngestionJobRecord, QueryResult, WorkspaceMembership


class BootstrapRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8)
    organization_slug: str
    organization_name: str
    workspace_slug: str
    workspace_name: str


class LoginRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'


class QueryRequest(BaseModel):
    workspace_id: str
    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    result: QueryResult


class DocumentResponse(BaseModel):
    document: DocumentRecord
    job: IngestionJobRecord | None = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentRecord]


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
    details: dict = Field(default_factory=dict)
