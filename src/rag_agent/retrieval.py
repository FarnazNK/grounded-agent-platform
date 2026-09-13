from __future__ import annotations

from collections import defaultdict

from rag_agent.config import Settings, get_settings
from rag_agent.schemas import RetrievedChunk
from rag_agent.store import RAGStore


class HybridRetriever:
    def __init__(self, store: RAGStore, settings: Settings | None = None) -> None:
        self._store = store
        self._settings = settings or get_settings()

    def retrieve(
        self, workspace_id: str, query: str, query_embedding: list[float]
    ) -> tuple[list[RetrievedChunk], dict[str, float]]:
        dense, dense_latency = self._store.dense_search(
            workspace_id, query_embedding, self._settings.top_k_dense
        )
        sparse, sparse_latency = self._store.lexical_search(
            workspace_id, query, self._settings.top_k_sparse
        )
        fused = reciprocal_rank_fusion([dense, sparse])
        filtered = [chunk for chunk in fused if chunk.fused_score >= self._settings.min_fused_score]
        return filtered[: self._settings.top_k_final], {
            "dense_ms": dense_latency,
            "sparse_ms": sparse_latency,
            "db_ms": dense_latency + sparse_latency,
        }


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    fused_scores: dict[str, float] = defaultdict(float)
    seen: dict[str, RetrievedChunk] = {}
    score_parts: dict[str, dict[str, float]] = defaultdict(dict)
    for results in result_lists:
        for rank, chunk in enumerate(results):
            fused_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
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
