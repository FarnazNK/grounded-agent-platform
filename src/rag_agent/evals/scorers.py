from __future__ import annotations

from dataclasses import dataclass

from rag_agent.evals.dataset import EvalCase
from rag_agent.schemas import QueryResult


@dataclass
class CaseMetrics:
    precision: float
    recall: float
    hit_rate: float
    mrr: float
    answer_relevance: float
    groundedness: float
    citation_correctness: float
    hallucination_rate: float
    latency_ms: float
    total_tokens: int
    estimated_cost_usd: float


def score_case(case: EvalCase, result: QueryResult) -> CaseMetrics:
    expected = case.expected_sources
    retrieved = [chunk.source_name for chunk in result.chunks]
    retrieved_set = set(retrieved)
    expected_set = set(expected)
    hits = expected_set & retrieved_set
    precision = len(hits) / max(1, len(retrieved_set))
    recall = len(hits) / max(1, len(expected_set)) if expected_set else 1.0
    hit_rate = 1.0 if hits else 0.0
    rank = next(
        (index + 1 for index, source in enumerate(retrieved) if source in expected_set), None
    )
    mrr = 1.0 / rank if rank else 0.0
    contains_hits = [
        snippet for snippet in case.expected_contains if snippet.lower() in result.answer.lower()
    ]
    answer_relevance = (
        len(contains_hits) / max(1, len(case.expected_contains)) if case.expected_contains else 1.0
    )
    expected_citations = set(case.expected_citations or expected)
    actual_citations = set(result.citations)
    citation_correctness = (
        len(expected_citations & actual_citations) / max(1, len(expected_citations))
        if expected_citations
        else 1.0
    )
    groundedness = 1.0 if result.grounded else 0.0
    hallucination_rate = 0.0 if result.grounded else 1.0
    return CaseMetrics(
        precision=precision,
        recall=recall,
        hit_rate=hit_rate,
        mrr=mrr,
        answer_relevance=answer_relevance,
        groundedness=groundedness,
        citation_correctness=citation_correctness,
        hallucination_rate=hallucination_rate,
        latency_ms=result.latency.total_ms,
        total_tokens=result.usage.total_tokens,
        estimated_cost_usd=result.usage.estimated_cost_usd,
    )
