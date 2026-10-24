from __future__ import annotations

from collections import defaultdict

from rag_agent.config import Settings, get_settings
from rag_agent.rerank import Reranker
from rag_agent.schemas import RetrievedChunk
from rag_agent.store import RAGStore


class HybridRetriever:
    def __init__(
        self,
        store: RAGStore,
        settings: Settings | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self._settings = settings or get_settings()
        self._reranker = reranker

    def retrieve(
        self, workspace_id: str, query: str, query_embedding: list[float]
    ) -> tuple[list[RetrievedChunk], dict[str, float]]:
        settings = self._settings
        # With a reranker, pull a wider candidate pool from each retriever so
        # the second stage has something to reorder.
        pool = settings.rerank_candidates if self._reranker else 0
        dense, dense_latency = self._store.dense_search(
            workspace_id, query_embedding, max(settings.top_k_dense, pool)
        )
        sparse, sparse_latency = self._store.lexical_search(
            workspace_id, query, max(settings.top_k_sparse, pool)
        )
        fused = reciprocal_rank_fusion(
            [dense, sparse],
            weights=[settings.dense_retrieval_weight, settings.sparse_retrieval_weight],
        )
        filtered = [chunk for chunk in fused if chunk.fused_score >= settings.min_fused_score]
        metrics = {
            "dense_ms": dense_latency,
            "sparse_ms": sparse_latency,
            "db_ms": dense_latency + sparse_latency,
            "rerank_ms": 0.0,
            "rerank_prompt_tokens": 0.0,
            "rerank_completion_tokens": 0.0,
        }
        if self._reranker is not None and filtered:
            candidates = filtered[: settings.rerank_candidates]
            filtered, stats = self._reranker.rerank(query, candidates)
            metrics["rerank_ms"] = stats.latency_ms
            metrics["rerank_prompt_tokens"] = float(stats.prompt_tokens)
            metrics["rerank_completion_tokens"] = float(stats.completion_tokens)
        return filtered[: settings.top_k_final], metrics


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must match the number of result lists")
    fused_scores: dict[str, float] = defaultdict(float)
    seen: dict[str, RetrievedChunk] = {}
    score_parts: dict[str, dict[str, float]] = defaultdict(dict)
    for list_index, results in enumerate(result_lists):
        weight = weights[list_index]
        for rank, chunk in enumerate(results):
            fused_scores[chunk.chunk_id] += weight / (k + rank + 1)
            seen.setdefault(chunk.chunk_id, chunk)
            if chunk.vector_score is not None:
                score_parts[chunk.chunk_id]["vector_score"] = chunk.vector_score
            if chunk.lexical_score is not None:
                score_parts[chunk.chunk_id]["lexical_score"] = chunk.lexical_score
    if not fused_scores:
        return []
    max_score = max(fused_scores.values())
    out: list[RetrievedChunk] = []
    for chunk_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
        out.append(
            seen[chunk_id].model_copy(
                update={"fused_score": score / max_score, **score_parts[chunk_id]}
            )
        )
    return out
