from __future__ import annotations

import re
from dataclasses import dataclass

from rag_agent.evals.dataset import EvalCase
from rag_agent.schemas import QueryResult

# Phrases a model uses when it (correctly) says the documents don't cover a
# question. Kept broad on purpose; the LLM judge is the stricter check.
_ABSTAIN_PATTERNS = re.compile(
    r"(could not find|couldn't find|can't find|cannot find|no (relevant )?information|"
    r"not (covered|mentioned|specified|included|available|in the (provided )?(context|documents))|"
    r"(does|do) not (contain|cover|mention|specify|say)|"
    r"(doesn't|don't) (contain|cover|mention|specify|say|have)|"
    r"isn't (covered|mentioned|specified)|no (policy|document) (on|about|for)|unable to answer)",
    re.I,
)


@dataclass
class CaseMetrics:
    # Answerable: the case names the documents that hold the answer. Retrieval
    # metrics are only meaningful for these.
    answerable: bool
    precision: float
    recall: float
    hit_rate: float
    mrr: float
    answer_relevance: float
    groundedness: float
    citation_correctness: float
    hallucination_rate: float
    # Cases tagged unanswerable only: did the system decline instead of
    # answering from unrelated documents? None for every other case.
    abstained: bool | None
    latency_ms: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


# Cases with these tags have no answer in the corpus; the right behavior is to
# say so. (Prompt-injection cases also lack expected sources but may expect a
# normal answer, so they aren't scored for abstention.)
ABSTAIN_TAGS = frozenset({"hallucination", "honesty", "out_of_scope"})


def should_abstain(case: EvalCase) -> bool:
    return not case.expected_sources and bool(ABSTAIN_TAGS & set(case.tags))


def is_abstention(result: QueryResult) -> bool:
    return result.refusal or not result.citations or bool(_ABSTAIN_PATTERNS.search(result.answer))


def score_case(case: EvalCase, result: QueryResult) -> CaseMetrics:
    expected = case.expected_sources
    answerable = bool(expected)
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
    answer_without_citations = re.sub(r"\[source:\s*[^\]]+\]", "", result.answer, flags=re.I)
    contains_hits = [
        snippet
        for snippet in case.expected_contains
        if snippet.lower() in answer_without_citations.lower()
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
        answerable=answerable,
        precision=precision,
        recall=recall,
        hit_rate=hit_rate,
        mrr=mrr,
        answer_relevance=answer_relevance,
        groundedness=groundedness,
        citation_correctness=citation_correctness,
        hallucination_rate=hallucination_rate,
        abstained=is_abstention(result) if should_abstain(case) else None,
        latency_ms=result.latency.total_ms,
        total_tokens=result.usage.total_tokens,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        estimated_cost_usd=result.usage.estimated_cost_usd,
    )
