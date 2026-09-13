from __future__ import annotations

from rag_agent.retrieval import reciprocal_rank_fusion
from rag_agent.schemas import RetrievedChunk


def chunk(
    chunk_id: str,
    source_name: str,
    fused_score: float = 0.1,
    vector_score: float | None = None,
    lexical_score: float | None = None,
):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        source_name=source_name,
        content="content",
        fused_score=fused_score,
        vector_score=vector_score,
        lexical_score=lexical_score,
    )


def test_rrf_boosts_overlap():
    first = [chunk("a", "a.md", vector_score=0.9), chunk("b", "b.md", vector_score=0.8)]
    second = [chunk("a", "a.md", lexical_score=0.7), chunk("c", "c.md", lexical_score=0.6)]
    fused = reciprocal_rank_fusion([first, second])
    assert fused[0].chunk_id == "a"
    assert fused[0].vector_score == 0.9
    assert fused[0].lexical_score == 0.7
