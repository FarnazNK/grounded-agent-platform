from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_agent.config import Settings
from rag_agent.llm import DeterministicLLMProvider, LLMResult
from rag_agent.rerank import (
    LexicalReranker,
    LLMReranker,
    apply_ranking,
    build_reranker,
    parse_ranking,
    stem,
)
from rag_agent.retrieval import HybridRetriever
from rag_agent.schemas import RetrievedChunk, UsageInfo


def chunk(source: str, content: str, fused: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=source, document_id=source, source_name=source, content=content, fused_score=fused
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("meetings", "meeting"),
        ("policies", "policy"),
        ("required", "require"),
        ("stages", "stage"),
        ("days", "day"),
    ],
)
def test_stem_maps_word_forms_together(a, b):
    assert stem(a) == stem(b)


def test_stem_leaves_short_words_and_numbers_alone():
    assert stem("vpn") == "vpn"
    assert stem("2025") == "2025"
    assert stem("access") == "access"


def test_lexical_reranker_promotes_the_document_that_covers_the_query():
    candidates = [
        chunk("dress_code.md", "No formal dress code. Business casual for customers.", 1.0),
        chunk("acceptable_use.md", "Company devices are for company business.", 0.9),
        chunk("meeting_norms.md", "Every meeting has an agenda. Async-first.", 0.6),
    ]
    reranked, stats = LexicalReranker().rerank("What is our stance on meetings?", candidates)
    assert reranked[0].source_name == "meeting_norms.md"
    assert "rerank_score" in reranked[0].metadata
    assert stats.prompt_tokens == 0


def test_lexical_reranker_ignores_question_words():
    # "am" must not match "9 AM", and "do" carries no topic signal.
    candidates = [
        chunk("sick_leave_policy.md", "Notify your manager before 9 AM. Do not wait.", 1.0),
        chunk("onboarding.md", "Week 1: complete compliance training.", 0.95),
    ]
    reranked, _ = LexicalReranker().rerank("What am I to do in week 1?", candidates)
    assert reranked[0].source_name == "onboarding.md"


def test_lexical_reranker_keeps_order_when_nothing_matches():
    candidates = [chunk("a.md", "alpha", 1.0), chunk("b.md", "beta", 0.5)]
    reranked, _ = LexicalReranker().rerank("the and of", candidates)
    assert [c.source_name for c in reranked] == ["a.md", "b.md"]


def test_parse_ranking_handles_prose_and_bad_json():
    assert parse_ranking('Sure! {"ranking": [2, 0, 1]}') == [2, 0, 1]
    assert parse_ranking("no json here") == []
    assert parse_ranking('{"ranking": "not a list"}') == []
    assert parse_ranking('{"ranking": [1, "x", 0]}') == [1, 0]


def test_apply_ranking_ignores_unknown_ids_and_keeps_missing_ones():
    chunks = [chunk(f"{i}.md", "x", 1.0) for i in range(4)]
    ordered = apply_ranking(chunks, [3, 9, 3, 1])
    assert [c.source_name for c in ordered] == ["3.md", "1.md", "0.md", "2.md"]


class FakeCompletion:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return LLMResult(
            text=self.text,
            usage=UsageInfo(prompt_tokens=120, completion_tokens=8, total_tokens=128),
            latency_ms=1.0,
        )


def test_llm_reranker_applies_model_order_and_reports_tokens():
    provider = FakeCompletion('{"ranking": [1, 0]}')
    chunks = [chunk("a.md", "alpha", 1.0), chunk("b.md", "beta", 0.9)]
    reranked, stats = LLMReranker(provider).rerank("q", chunks)
    assert [c.source_name for c in reranked] == ["b.md", "a.md"]
    assert (stats.prompt_tokens, stats.completion_tokens) == (120, 8)
    system, user = provider.calls[0]
    assert "untrusted data" in system
    assert "[0] (a.md)" in user and "[1] (b.md)" in user


def test_llm_reranker_falls_back_to_input_order_on_errors():
    chunks = [chunk("a.md", "alpha", 1.0), chunk("b.md", "beta", 0.9)]
    reranked, _ = LLMReranker(FakeCompletion(error=TimeoutError())).rerank("q", chunks)
    assert [c.source_name for c in reranked] == ["a.md", "b.md"]


def test_build_reranker_choices():
    assert build_reranker(Settings(reranker="none"), DeterministicLLMProvider()) is None
    assert isinstance(
        build_reranker(Settings(reranker="lexical"), DeterministicLLMProvider()), LexicalReranker
    )
    with pytest.raises(ValueError):
        build_reranker(Settings.model_construct(reranker="llm"), DeterministicLLMProvider())


def test_llm_reranker_requires_a_remote_provider_in_settings():
    with pytest.raises(ValueError, match="APP_RERANKER=llm"):
        Settings(reranker="llm")


class StubStore:
    def __init__(self) -> None:
        self.limits: list[int] = []

    def dense_search(self, workspace_id, embedding, limit):
        self.limits.append(limit)
        return [chunk(f"d{i}.md", "dense", 1.0) for i in range(limit)], 1.0

    def lexical_search(self, workspace_id, query, limit):
        self.limits.append(limit)
        return [chunk(f"s{i}.md", "sparse", 1.0) for i in range(limit)], 1.0


def test_retriever_widens_the_pool_only_when_reranking():
    settings = Settings(top_k_dense=6, top_k_sparse=6, top_k_final=4, rerank_candidates=12)
    store = StubStore()
    plain = HybridRetriever(store, settings.model_copy(update={"reranker": "none"}), None)
    results, _ = plain.retrieve("w", "q", [0.0])
    assert store.limits == [6, 6]
    assert len(results) == 4

    store.limits.clear()
    reranked = HybridRetriever(store, settings, LexicalReranker())
    results, metrics = reranked.retrieve("w", "q", [0.0])
    assert store.limits == [12, 12]
    assert len(results) == 4
    assert metrics["rerank_ms"] >= 0.0


def test_service_cache_key_separates_rerankers():
    from rag_agent.evals.dataset import load_dataset
    from rag_agent.evals.runner import build_eval_service

    dataset = load_dataset("data/eval_datasets/quality.yaml")
    service, user_id, workspace_id = build_eval_service(
        dataset, Settings(app_env="test", retrieval_cache_ttl_seconds=60)
    )
    service.retrieval_cache = SimpleNamespace(  # type: ignore[assignment]
        keys=[],
        get=lambda key: None,
        set=lambda key, value: service.retrieval_cache.keys.append(key),
    )
    service.query(user_id=user_id, workspace_id=workspace_id, query="PTO", request_id="1")
    assert service.retrieval_cache.keys[0].endswith(":lexical")
