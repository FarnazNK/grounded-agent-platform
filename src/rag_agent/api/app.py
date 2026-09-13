from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_agent.api.metrics import (
    ACTIVE_REQUESTS,
    CACHE_HITS,
    CACHE_MISSES,
    DB_LATENCY,
    ERRORS,
    INGESTION_FAILURES,
    INGESTION_JOBS,
    LLM_LATENCY,
    REQUESTS,
    REQUEST_LATENCY,
    RETRIEVAL_GROUNDEDNESS,
    RETRIEVAL_LATENCY,
    TOKENS,
    render_metrics,
)
from rag_agent.api.schemas import (
    BootstrapRequest,
    DocumentResponse,
    DocumentsResponse,
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
from rag_agent.errors import AppError, AuthenticationRequiredError
from rag_agent.observability import maybe_enable_langsmith, request_context
from rag_agent.postgres_store import PostgresRAGStore
from rag_agent.service import RAGService

security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if not hasattr(app.state, 'service'):
        app.state.settings = get_settings()
        app.state.langsmith_enabled = maybe_enable_langsmith()
        app.state.store = PostgresRAGStore(get_session_factory())
        app.state.service = RAGService(app.state.store, settings=app.state.settings)
    yield


def create_app(service: RAGService | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.api_title, version='1.0.0', lifespan=_lifespan)
    if service is not None:
        app.state.settings = settings
        app.state.langsmith_enabled = False
        app.state.store = service.store
        app.state.service = service

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        ERRORS.labels(code=exc.code.value).inc()
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.middleware('http')
    async def metrics_middleware(request: Request, call_next):
        request_id = request.headers.get('x-request-id', str(uuid4()))
        start = time.perf_counter()
        ACTIVE_REQUESTS.inc()
        endpoint = request.url.path
        with request_context(request_id=request_id, path=endpoint):
            try:
                response = await call_next(request)
            except Exception:
                REQUESTS.labels(endpoint=endpoint, status='500').inc()
                raise
            finally:
                ACTIVE_REQUESTS.dec()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        response.headers['X-Request-ID'] = request_id
        return response

    def get_service() -> RAGService:
        return app.state.service

    def current_user(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        if credentials is None:
            raise AuthenticationRequiredError()
        payload = decode_access_token(credentials.credentials)
        return svc.get_user(payload['sub'])

    @app.get('/health/live')
    async def live() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get('/health/ready')
    async def ready() -> dict[str, object]:
        if service is None:
            ping_database()
        docs, chunks = app.state.store.corpus_counts()
        return {'status': 'ok', 'documents': docs, 'chunks': chunks, 'langsmith_enabled': bool(app.state.langsmith_enabled)}

    @app.get('/metrics')
    async def metrics() -> Response:
        return Response(content=render_metrics(), media_type='text/plain; version=0.0.4')

    @app.post('/v1/auth/bootstrap', response_model=WorkspaceListResponse, responses={400: {'model': ErrorResponse}})
    async def bootstrap(req: BootstrapRequest, svc: Annotated[RAGService, Depends(get_service)]):
        membership = svc.bootstrap_admin(**req.model_dump())
        return WorkspaceListResponse(workspaces=[membership])

    @app.post('/v1/auth/login', response_model=TokenResponse, responses={401: {'model': ErrorResponse}})
    async def login(req: LoginRequest, svc: Annotated[RAGService, Depends(get_service)]):
        auth = svc.authenticate(req.email, req.password)
        return TokenResponse(access_token=auth.access_token)

    @app.get('/v1/auth/me', response_model=MeResponse)
    async def me(user=Depends(current_user)):
        return MeResponse(user_id=user.id, email=user.email)

    @app.get('/v1/workspaces', response_model=WorkspaceListResponse)
    async def list_workspaces(
        user=Depends(current_user),
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        return WorkspaceListResponse(workspaces=svc.list_workspaces(user.id))

    @app.post('/v1/documents/upload', response_model=DocumentResponse)
    async def upload_document(
        workspace_id: str,
        file: UploadFile = File(...),
        user=Depends(current_user),
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        try:
            document, job = svc.ingest_document(user_id=user.id, workspace_id=workspace_id, filename=file.filename or 'upload.txt', content_type=file.content_type, data=await file.read())
        except AppError as exc:
            if exc.details.get('job_id'):
                INGESTION_FAILURES.labels(reason=exc.code.value).inc()
            raise
        INGESTION_JOBS.labels(operation='upload', status=job.status.value).inc()
        return DocumentResponse(document=document, job=job)

    @app.get('/v1/documents', response_model=DocumentsResponse)
    async def list_documents(
        workspace_id: str,
        user=Depends(current_user),
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        return DocumentsResponse(documents=svc.list_documents(user_id=user.id, workspace_id=workspace_id))

    @app.get('/v1/documents/{document_id}', response_model=DocumentResponse)
    async def get_document(
        document_id: str,
        workspace_id: str,
        user=Depends(current_user),
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        return DocumentResponse(document=svc.get_document(user_id=user.id, workspace_id=workspace_id, document_id=document_id))

    @app.post('/v1/documents/{document_id}/reindex', response_model=DocumentResponse)
    async def reindex_document(
        document_id: str,
        workspace_id: str,
        user=Depends(current_user),
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        document, job = svc.reindex_document(user_id=user.id, workspace_id=workspace_id, document_id=document_id)
        INGESTION_JOBS.labels(operation='reindex', status=job.status.value).inc()
        return DocumentResponse(document=document, job=job)

    @app.delete('/v1/documents/{document_id}', response_model=DocumentResponse)
    async def delete_document(
        document_id: str,
        workspace_id: str,
        user=Depends(current_user),
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        document, job = svc.delete_document(user_id=user.id, workspace_id=workspace_id, document_id=document_id)
        INGESTION_JOBS.labels(operation='delete', status=job.status.value).inc()
        return DocumentResponse(document=document, job=job)

    @app.get('/v1/ingestions/{job_id}', response_model=IngestionJobResponse)
    async def get_ingestion_job(
        job_id: str,
        workspace_id: str,
        user=Depends(current_user),
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        return IngestionJobResponse(job=svc.get_ingestion_job(user_id=user.id, workspace_id=workspace_id, job_id=job_id))

    @app.post('/v1/query', response_model=QueryResponse)
    async def query(
        req: QueryRequest,
        request: Request,
        user=Depends(current_user),
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        request_id = request.headers.get('x-request-id', str(uuid4()))
        result = svc.query(user_id=user.id, workspace_id=req.workspace_id, query=req.query, request_id=request_id)
        if result.cached:
            CACHE_HITS.labels(cache='retrieval').inc()
        else:
            CACHE_MISSES.labels(cache='retrieval').inc()
        RETRIEVAL_LATENCY.observe(result.latency.retrieval_ms / 1000)
        LLM_LATENCY.observe(result.latency.llm_ms / 1000)
        DB_LATENCY.observe(result.latency.db_ms / 1000)
        TOKENS.labels(type='prompt').inc(result.usage.prompt_tokens)
        TOKENS.labels(type='completion').inc(result.usage.completion_tokens)
        RETRIEVAL_GROUNDEDNESS.set(1.0 if result.grounded else 0.0)
        return QueryResponse(result=result)

    @app.post('/v1/stream')
    async def stream(
        req: QueryRequest,
        request: Request,
        user=Depends(current_user),
        svc: Annotated[RAGService, Depends(get_service)],
    ):
        request_id = request.headers.get('x-request-id', str(uuid4()))

        async def event_generator():
            result = svc.query(user_id=user.id, workspace_id=req.workspace_id, query=req.query, request_id=request_id)
            yield f"event: retrieval\ndata: {json.dumps({'chunk_count': len(result.chunks)})}\n\n"
            yield f"event: answer\ndata: {json.dumps({'answer': result.answer, 'citations': result.citations})}\n\n"
            yield f"event: done\ndata: {json.dumps({'request_id': result.request_id, 'cached': result.cached})}\n\n"

        return StreamingResponse(event_generator(), media_type='text/event-stream')

    return app
