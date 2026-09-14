from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProjectRoot = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    app_env: Literal["dev", "staging", "production", "test"] = "dev"
    api_title: str = "RAG Agent API"

    database_url: str = "postgresql+psycopg://rag_agent@localhost:5432/rag_agent"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: Literal["HS256"] = "HS256"
    jwt_access_token_ttl_minutes: int = Field(default=60, ge=5, le=1440)
    enable_bootstrap_admin: bool = True

    llm_provider: Literal["anthropic", "openai", "deterministic"] = "deterministic"
    llm_model: str = "deterministic-rag"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_request_timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)
    llm_max_prompt_tokens: int = Field(default=4000, ge=128)
    llm_max_output_tokens: int = Field(default=800, ge=32)

    embedding_provider: Literal["openai", "deterministic"] = "deterministic"
    embedding_model: str = "deterministic-embedding"
    embedding_dimensions: Literal[256] = 256
    embedding_request_timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)

    top_k_dense: int = Field(default=6, ge=1, le=100)
    top_k_sparse: int = Field(default=6, ge=1, le=100)
    top_k_final: int = Field(default=4, ge=1, le=50)
    dense_retrieval_weight: float = Field(default=0.8, gt=0.0, le=5.0)
    sparse_retrieval_weight: float = Field(default=1.2, gt=0.0, le=5.0)
    min_fused_score: float = Field(default=0.15, ge=0.0, le=1.0)
    chunk_size_words: int = Field(default=180, ge=20, le=2000)
    chunk_overlap_words: int = Field(default=30, ge=0, le=1000)

    upload_max_bytes: int = Field(default=2_000_000, ge=1024, le=50_000_000)
    allowed_mime_types: tuple[str, ...] = (
        "text/plain",
        "text/markdown",
        "application/pdf",
        "application/json",
    )

    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=100_000)
    auth_rate_limit_requests_per_minute: int = Field(default=10, ge=1, le=1_000)
    retrieval_cache_ttl_seconds: int = Field(default=120, ge=0, le=86_400)
    embedding_cache_ttl_seconds: int = Field(default=3600, ge=0, le=604_800)

    log_level: str = "INFO"
    metrics_namespace: str = "rag_agent"
    langsmith_project: str = "rag-agent"

    eval_recall_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    eval_groundedness_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    eval_answer_relevance_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    eval_p95_latency_ms_threshold: float = Field(default=500.0, gt=0.0)

    benchmark_output_dir: Path = Field(default=ProjectRoot / "benchmarks" / "results")

    @model_validator(mode="after")
    def validate_runtime_configuration(self) -> Settings:
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        if self.top_k_final > self.top_k_dense + self.top_k_sparse:
            raise ValueError("top_k_final cannot exceed the total retrieval candidate count")
        if self.app_env in {"staging", "production"}:
            if self.jwt_secret == "change-me-in-production" or len(self.jwt_secret) < 32:
                raise ValueError(
                    "staging/production requires a JWT secret of at least 32 characters"
                )
            if self.enable_bootstrap_admin:
                raise ValueError("bootstrap admin must be disabled in staging/production")
        if self.llm_provider != "deterministic" and self.llm_model == "deterministic-rag":
            raise ValueError("set APP_LLM_MODEL when using a remote LLM provider")
        if (
            self.embedding_provider != "deterministic"
            and self.embedding_model == "deterministic-embedding"
        ):
            raise ValueError("set APP_EMBEDDING_MODEL when using a remote embedding provider")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()