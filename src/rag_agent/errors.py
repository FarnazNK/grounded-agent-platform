from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    AUTHENTICATION_REQUIRED = "authentication_required"
    AUTHORIZATION_FAILED = "authorization_failed"
    INVALID_UPLOAD = "invalid_upload"
    DUPLICATE_INGESTION = "duplicate_ingestion"
    DOCUMENT_NOT_FOUND = "document_not_found"
    INGESTION_NOT_FOUND = "ingestion_not_found"
    WORKSPACE_NOT_FOUND = "workspace_not_found"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    MALFORMED_LLM_OUTPUT = "malformed_llm_output"
    EMBEDDING_FAILURE = "embedding_failure"
    DATABASE_UNAVAILABLE = "database_unavailable"
    VALIDATION_FAILED = "validation_failed"


class AppError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        status_code: int,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


class AuthenticationRequiredError(AppError):
    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(ErrorCode.AUTHENTICATION_REQUIRED, message, status_code=401)


class AuthorizationFailedError(AppError):
    def __init__(self, message: str = "Forbidden") -> None:
        super().__init__(ErrorCode.AUTHORIZATION_FAILED, message, status_code=403)


class InvalidUploadError(AppError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.INVALID_UPLOAD, message, status_code=400, details=details)


class DuplicateIngestionError(AppError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            ErrorCode.DUPLICATE_INGESTION,
            message,
            status_code=409,
            details=details,
        )


class DocumentNotFoundError(AppError):
    def __init__(self, document_id: str) -> None:
        super().__init__(
            ErrorCode.DOCUMENT_NOT_FOUND,
            f"Document {document_id} was not found.",
            status_code=404,
        )


class IngestionNotFoundError(AppError):
    def __init__(self, job_id: str) -> None:
        super().__init__(
            ErrorCode.INGESTION_NOT_FOUND,
            f"Ingestion job {job_id} was not found.",
            status_code=404,
        )


class WorkspaceNotFoundError(AppError):
    def __init__(self, workspace_id: str) -> None:
        super().__init__(
            ErrorCode.WORKSPACE_NOT_FOUND,
            f"Workspace {workspace_id} was not found.",
            status_code=404,
        )


class TokenLimitExceededError(AppError):
    def __init__(self, message: str = "Prompt exceeds configured token limit.") -> None:
        super().__init__(ErrorCode.TOKEN_LIMIT_EXCEEDED, message, status_code=413)


class ProviderTimeoutError(AppError):
    def __init__(self, message: str = "Provider call timed out.") -> None:
        super().__init__(ErrorCode.LLM_TIMEOUT, message, status_code=504, retryable=True)


class ProviderRateLimitError(AppError):
    def __init__(self, message: str = "Provider rate limit exceeded.") -> None:
        super().__init__(ErrorCode.LLM_RATE_LIMIT, message, status_code=429, retryable=True)


class MalformedLLMOutputError(AppError):
    def __init__(self, message: str = "Provider returned malformed output.") -> None:
        super().__init__(ErrorCode.MALFORMED_LLM_OUTPUT, message, status_code=502)


class EmbeddingProviderError(AppError):
    def __init__(self, message: str = "Embedding provider failed.") -> None:
        super().__init__(ErrorCode.EMBEDDING_FAILURE, message, status_code=502, retryable=True)


class DatabaseUnavailableError(AppError):
    def __init__(self, message: str = "Database is temporarily unavailable.") -> None:
        super().__init__(ErrorCode.DATABASE_UNAVAILABLE, message, status_code=503, retryable=True)


class RequestValidationError(AppError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.VALIDATION_FAILED, message, status_code=422, details=details)
