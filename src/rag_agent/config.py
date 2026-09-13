from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
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
    jwt_algorithm: str = "HS256"
    jwt_access_token_ttl_minutes: int = 60
    enable_bootstrap_admin: bool = True

    llm_provider: Literal["anthropic", "openai", "deterministic"] = "deterministic"
    llm_model: str = "deterministic-rag"
    llm_temperature: float = 0.0
    llm_request_timeout_seconds: float = 20.0
    llm_max_prompt_tokens: int = 4000
    llm_max_output_tokens: int = 800

    embedding_provider: Literal["openai", "deterministic"] = "deterministic"
    embedding_model: str = "deterministic-embedding"
    embedding_dimensions: int = 16
    embedding_request_timeout_seconds: float = 20.0

    top_k_dense: int = 6
    top_k_sparse: int = 6
    top_k_final: int = 4
    min_fused_score: float = 0.15
    chunk_size_words: int = 180
    chunk_overlap_words: int = 30

    upload_max_bytes: int = 2_000_000
    allowed_mime_types: tuple[str, ...] = (
        "text/plain",
        "text/markdown",
        "application/pdf",
        "application/json",
    )

    request_timeout_seconds: float = 30.0
    rate_limit_requests_per_minute: int = 60
    retrieval_cache_ttl_seconds: int = 120
    embedding_cache_ttl_seconds: int = 3600

    log_level: str = "INFO"
    metrics_namespace: str = "rag_agent"
    langsmith_project: str = "rag-agent"

    eval_recall_threshold: float = 0.75
    eval_groundedness_threshold: float = 0.90
    eval_p95_latency_ms_threshold: float = 500.0

    benchmark_output_dir: Path = Field(default=ProjectRoot / "benchmarks" / "results")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
