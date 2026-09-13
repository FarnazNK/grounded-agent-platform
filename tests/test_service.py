from __future__ import annotations

import pytest

from rag_agent.config import Settings
from rag_agent.embeddings import EmbeddingService
from rag_agent.errors import EmbeddingProviderError, MalformedLLMOutputError, ProviderTimeoutError, TokenLimitExceededError
from rag_agent.llm import LLMResult
from rag_agent.schemas import UsageInfo
from rag_agent.service import RAGService
from rag_agent.store import InMemoryRAGStore
from tests.conftest import build_seeded_service


class FailingEmbeddingProvider:
    model_name = 'failing-embedding'

    def embed_documents(self, texts):
        raise EmbeddingProviderError('embed failed')

    def embed_query(self, text):
        raise EmbeddingProviderError('embed failed')


class TimeoutLLM:
    def answer_question(self, query, chunks):
        raise ProviderTimeoutError()


class BlankLLM:
    def answer_question(self, query, chunks):
        return LLMResult(text='   ', usage=UsageInfo(total_tokens=1), latency_ms=1)


class CountingLLM:
    def __init__(self):
        self.calls = 0

    def answer_question(self, query, chunks):
        self.calls += 1
        return LLMResult(
            text=f'answer {self.calls} [source: {chunks[0].source_name}]',
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_ms=1,
        )


def test_query_is_tenant_scoped(seeded_app):
    result = seeded_app.service.query(
        user_id=seeded_app.bob_id,
        workspace_id=seeded_app.bob_workspace_id,
        query='What is the PTO policy?',
        request_id='req',
    )
    assert {chunk.source_name for chunk in result.chunks} == {'beta_policy.md'}
    assert 'beta_policy.md' in result.answer


def test_cache_hit_and_invalidation(seeded_app):
    llm = CountingLLM()
    seeded_app.service.llm = llm
    first = seeded_app.service.query(
        user_id=seeded_app.alice_id,
        workspace_id=seeded_app.alice_workspace_id,
        query='What is the PTO policy?',
        request_id='one',
    )
    second = seeded_app.service.query(
        user_id=seeded_app.alice_id,
        workspace_id=seeded_app.alice_workspace_id,
        query='What is the PTO policy?',
        request_id='two',
    )
    assert not first.cached
    assert second.cached
    assert llm.calls == 1

    document = seeded_app.service.list_documents(user_id=seeded_app.alice_id, workspace_id=seeded_app.alice_workspace_id)[0]
    seeded_app.service.reindex_document(
        user_id=seeded_app.alice_id,
        workspace_id=seeded_app.alice_workspace_id,
        document_id=document.id,
    )
    third = seeded_app.service.query(
        user_id=seeded_app.alice_id,
        workspace_id=seeded_app.alice_workspace_id,
        query='What is the PTO policy?',
        request_id='three',
    )
    assert not third.cached
    assert llm.calls == 2


def test_embedding_failure_marks_ingestion_failed():
    settings = Settings(app_env='test', llm_provider='deterministic', embedding_provider='deterministic', jwt_secret='test-secret')
    store = InMemoryRAGStore()
    service = RAGService(store, settings=settings, embedding_service=EmbeddingService(FailingEmbeddingProvider()))
    membership = service.bootstrap_admin(
        email='owner@example.com',
        password='password123',
        organization_slug='org',
        organization_name='Org',
        workspace_slug='workspace',
        workspace_name='Workspace',
    )
    user_id = store.get_user_by_email('owner@example.com').id
    with pytest.raises(EmbeddingProviderError):
        service.ingest_document(
            user_id=user_id,
            workspace_id=membership.workspace.id,
            filename='broken.md',
            content_type='text/markdown',
            data=b'hello world',
        )
    document = service.list_documents(user_id=user_id, workspace_id=membership.workspace.id)[0]
    assert document.status == 'failed'


def test_llm_timeout_bubbles_up(seeded_app):
    seeded_app.service.llm = TimeoutLLM()
    with pytest.raises(ProviderTimeoutError):
        seeded_app.service.query(
            user_id=seeded_app.alice_id,
            workspace_id=seeded_app.alice_workspace_id,
            query='What is the PTO policy?',
            request_id='req',
        )


def test_malformed_llm_output_is_rejected(seeded_app):
    seeded_app.service.llm = BlankLLM()
    with pytest.raises(MalformedLLMOutputError):
        seeded_app.service.query(
            user_id=seeded_app.alice_id,
            workspace_id=seeded_app.alice_workspace_id,
            query='What is the PTO policy?',
            request_id='req',
        )


def test_token_limit_exceeded():
    seeded = build_seeded_service()
    seeded.service.settings = seeded.service.settings.model_copy(update={'llm_max_prompt_tokens': 4})
    with pytest.raises(TokenLimitExceededError):
        seeded.service.query(
            user_id=seeded.alice_id,
            workspace_id=seeded.alice_workspace_id,
            query='this query is intentionally too long',
            request_id='req',
        )
