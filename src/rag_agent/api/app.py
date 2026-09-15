from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import partial
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from rag_agent.api.metrics import (
    ACTIVE_REQUESTS,
    CACHE_HITS,
    CACHE_MISSES,
    DB_LATENCY,
    ERRORS,
    INGESTION_FAILURES,
    INGESTION_JOBS,
    LLM_LATENCY,
    REQUEST_LATENCY,
    REQUESTS,
    RETRIEVAL_GROUNDEDNESS,
    RETRIEVAL_LATENCY,
    TOKENS,
    render_metrics,
)
from rag_agent.api.schemas import (
    BootstrapRequest,
    DocumentResponse,
    DocumentsResponse,
    DocumentView,
    ErrorResponse,
    IngestionJobResponse,
    LoginRequest,
    MeResponse,
    QueryRequest,
    QueryResponse,
    TokenResponse,
    WorkspaceListResponse,
)
from rag_agent.auth import decode_access_token
from rag_agent.config import get_settings
from rag_agent.db import get_session_factory, ping_database
from rag_agent.errors import AppError, AuthenticationRequiredError, InvalidUploadError
from rag_agent.observability import maybe_enable_langsmith, request_context
from rag_agent.service import RAGService

security = HTTPBearer(auto_error=False)
_auth_windows: dict[str, deque[float]] = defaultdict(deque)


def _enforce_auth_rate_limit(request: Request, limit: int) -> None:
    now = time.monotonic()
    key = request.client.host if request.client else "unknown"
    events = _auth_windows[key]
    while events and now - events[0] >= 60:
        events.popleft()
    if len(events) >= limit:
        from rag_agent.errors import RateLimitExceededError

        raise RateLimitExceededError("Authentication rate limit exceeded.")
    events.append(now)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if not hasattr(app.state, "service"):
        # Import lazily so in-memory/unit-test usage does not require pgvector at import time.
        from rag_agent.postgres_store import PostgresRAGStore

        app.state.settings = get_settings()
        app.state.langsmith_enabled = maybe_enable_langsmith()
        app.state.store = PostgresRAGStore(get_session_factory())
        app.state.service = RAGService(app.state.store, settings=app.state.settings)
    yield


def create_app(service: RAGService | None = None) -> FastAPI:
    settings = service.settings if service is not None else get_settings()
    app = FastAPI(title=settings.api_title, version="1.0.0", lifespan=_lifespan)
    if service is not None:
        app.state.settings = settings
        app.state.langsmith_enabled = False
        app.state.store = service.store
        app.state.service = service

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        ERRORS.labels(code=exc.code.value).inc()
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        start = time.perf_counter()
        status = "500"
        endpoint = request.url.path
        response: Response | None = None
        ACTIVE_REQUESTS.inc()
        try:
            with request_context(request_id=request_id, path=request.url.path):
                response = await call_next(request)
                status = str(response.status_code)
                route = request.scope.get("route")
                endpoint = getattr(route, "path", request.url.path)
                response.headers["X-Request-ID"] = request_id
                return response
        finally:
            ACTIVE_REQUESTS.dec()
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
            REQUESTS.labels(endpoint=endpoint, status=status).inc()

    def get_service() -> RAGService:
        return app.state.service

    def current_user(
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
        svc: RAGService = Depends(get_service),
    ):
        if credentials is None:
            raise AuthenticationRequiredError()
        payload = decode_access_token(credentials.credentials, app.state.settings)
        return svc.get_user(payload["sub"])

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": "RAG Agent API",
            "status": "online",
            "docs": "/docs",
            "liveness": "/health/live",
            "readiness": "/health/ready",
        }

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, object]:
        if service is None:
            ping_database()
        if app.state.settings.app_env == "production":
            app.state.store.corpus_counts()
            return {"status": "ok"}

        docs, chunks = app.state.store.corpus_counts()
        return {
            "status": "ok",
            "documents": docs,
            "chunks": chunks,
            "langsmith_enabled": bool(app.state.langsmith_enabled),
        }

    @app.get("/metrics")
    def metrics(
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
        svc: RAGService = Depends(get_service),
    ) -> Response:
        if app.state.settings.app_env == "production":
            if credentials is None:
                raise AuthenticationRequiredError()
            payload = decode_access_token(credentials.credentials, app.state.settings)
            svc.get_user(payload["sub"])
        return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")

    @app.post(
        "/v1/auth/bootstrap",
        response_model=WorkspaceListResponse,
        responses={400: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
    )
    def bootstrap(req: BootstrapRequest, svc: RAGService = Depends(get_service)):
        membership = svc.bootstrap_admin(**req.model_dump())
        return WorkspaceListResponse(workspaces=[membership])

    @app.post(
        "/v1/auth/login",
        response_model=TokenResponse,
        responses={401: {"model": ErrorResponse}},
    )
    def login(
        req: LoginRequest,
        request: Request,
        svc: RAGService = Depends(get_service),
    ):
        _enforce_auth_rate_limit(
            request,
            app.state.settings.auth_rate_limit_requests_per_minute,
        )
        auth = svc.authenticate(req.email, req.password)
        return TokenResponse(access_token=auth.access_token)

    @app.get("/v1/auth/me", response_model=MeResponse)
    def me(user=Depends(current_user)):
        return MeResponse(user_id=user.id, email=user.email)

    @app.get("/v1/workspaces", response_model=WorkspaceListResponse)
    def list_workspaces(
        user=Depends(current_user),
        svc: RAGService = Depends(get_service),
    ):
        return WorkspaceListResponse(workspaces=svc.list_workspaces(user.id))

    @app.post("/v1/documents/upload", response_model=DocumentResponse)
    async def upload_document(
        workspace_id: str,
        file: UploadFile = File(...),
        user=Depends(current_user),
        svc: RAGService = Depends(get_service),
    ):
        max_bytes = app.state.settings.upload_max_bytes
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise InvalidUploadError(
                "Uploaded file exceeds size limit.",
                details={"max_bytes": max_bytes, "actual_bytes": len(data)},
            )
        try:
            document, job = await run_in_threadpool(
                partial(
                    svc.ingest_document,
                    user_id=user.id,
                    workspace_id=workspace_id,
                    filename=file.filename or "upload.txt",
                    content_type=file.content_type,
                    data=data,
                )
            )
        except AppError as exc:
            INGESTION_FAILURES.labels(reason=exc.code.value).inc()
            raise
        INGESTION_JOBS.labels(operation="upload", status=job.status.value).inc()
        return DocumentResponse(document=DocumentView.model_validate(document), job=job)

    @app.get("/v1/documents", response_model=DocumentsResponse)
    def list_documents(
        workspace_id: str,
        user=Depends(current_user),
        svc: RAGService = Depends(get_service),
    ):
        documents = svc.list_documents(user_id=user.id, workspace_id=workspace_id)
        return DocumentsResponse(documents=[DocumentView.model_validate(doc) for doc in documents])

    @app.get("/v1/documents/{document_id}", response_model=DocumentResponse)
    def get_document(
        document_id: str,
        workspace_id: str,
        user=Depends(current_user),
        svc: RAGService = Depends(get_service),
    ):
        document = svc.get_document(
            user_id=user.id,
            workspace_id=workspace_id,
            document_id=document_id,
        )
        return DocumentResponse(document=DocumentView.model_validate(document))

    @app.post("/v1/documents/{document_id}/reindex", response_model=DocumentResponse)
    def reindex_document(
        document_id: str,
        workspace_id: str,
        user=Depends(current_user),
        svc: RAGService = Depends(get_service),
    ):
        document, job = svc.reindex_document(
            user_id=user.id,
            workspace_id=workspace_id,
            document_id=document_id,
        )
        INGESTION_JOBS.labels(operation="reindex", status=job.status.value).inc()
        return DocumentResponse(document=DocumentView.model_validate(document), job=job)

    @app.delete("/v1/documents/{document_id}", response_model=DocumentResponse)
    def delete_document(
        document_id: str,
        workspace_id: str,
        user=Depends(current_user),
        svc: RAGService = Depends(get_service),
    ):
        document, job = svc.delete_document(
            user_id=user.id,
            workspace_id=workspace_id,
            document_id=document_id,
        )
        INGESTION_JOBS.labels(operation="delete", status=job.status.value).inc()
        return DocumentResponse(document=DocumentView.model_validate(document), job=job)

    @app.get("/v1/ingestions/{job_id}", response_model=IngestionJobResponse)
    def get_ingestion_job(
        job_id: str,
        workspace_id: str,
        user=Depends(current_user),
        svc: RAGService = Depends(get_service),
    ):
        job = svc.get_ingestion_job(
            user_id=user.id,
            workspace_id=workspace_id,
            job_id=job_id,
        )
        return IngestionJobResponse(job=job)

    @app.post("/v1/query", response_model=QueryResponse)
    def query(
        req: QueryRequest,
        request: Request,
        user=Depends(current_user),
        svc: RAGService = Depends(get_service),
    ):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        result = svc.query(
            user_id=user.id,
            workspace_id=req.workspace_id,
            query=req.query,
            request_id=request_id,
        )
        if result.cached:
            CACHE_HITS.labels(cache="retrieval").inc()
        else:
            CACHE_MISSES.labels(cache="retrieval").inc()
        RETRIEVAL_LATENCY.observe(result.latency.retrieval_ms / 1000)
        LLM_LATENCY.observe(result.latency.llm_ms / 1000)
        DB_LATENCY.observe(result.latency.db_ms / 1000)
        TOKENS.labels(type="prompt").inc(result.usage.prompt_tokens)
        TOKENS.labels(type="completion").inc(result.usage.completion_tokens)
        RETRIEVAL_GROUNDEDNESS.set(1.0 if result.grounded else 0.0)
        return QueryResponse(result=result)

    return app
