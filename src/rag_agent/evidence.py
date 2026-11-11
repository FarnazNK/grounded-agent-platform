"""Evidence gate: refuse before generation when retrieval found nothing relevant.

Fused RRF scores depend only on rank, so the top chunk always scores well even for a
question the corpus does not cover. This gate measures relevance directly instead: the
share of the question's content terms that appear in the retrieved chunks, weighted by
how rare each term is in the workspace (IDF). A term that appears nowhere in the workspace,
such as "weather" in an HR corpus, carries the most weight.

Lexical evidence cannot tell a paraphrase ("vacation" for "PTO") from an unsupported
topic ("sabbatical" when there is no sabbatical policy). The default threshold is therefore
conservative: it refuses questions with essentially no overlap, and leaves borderline
cases to the model, whose prompt tells it to say when the context lacks the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rag_agent.schemas import RetrievedChunk
from rag_agent.store import _terms, stem, stem_matches

# Question and filler words that say nothing about the topic being asked about.
GENERIC_QUESTION_TERMS = frozenset(
    {
        "about", "any", "being", "can", "could", "did", "do", "does", "far", "get", "give",
        "how", "know", "long", "many", "me", "much", "my", "our", "please", "tell", "there",
        "us", "we", "what", "when", "where", "which", "who", "whom", "why", "would", "you",
        "your",
    }
)  # fmt: skip

REFUSAL_ANSWER = (
    "I couldn't find information about this in the workspace documents, so I can't answer "
    "it without guessing."
)


@dataclass(frozen=True)
class EvidenceAssessment:
    coverage: float
    query_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)


def topic_terms(query: str) -> list[str]:
    """Stemmed content terms of the question, in order, without duplicates or question words."""
    return list(
        dict.fromkeys(stem(term) for term in _terms(query) if term not in GENERIC_QUESTION_TERMS)
    )


def assess_evidence(
    query: str,
    chunks: list[RetrievedChunk],
    document_frequencies: dict[str, int],
    total_chunks: int,
) -> EvidenceAssessment:
    """IDF-weighted share of the question's topic terms found in the retrieved chunks."""
    terms = topic_terms(query)
    if not terms:
        # Nothing topical to check (e.g. "what?"); let the normal pipeline decide.
        return EvidenceAssessment(coverage=1.0)
    words = set(_terms(" ".join(chunk.content for chunk in chunks)))
    present = {term for term in terms if any(stem_matches(term, word) for word in words)}
    weights = {
        term: math.log((total_chunks + 1) / (document_frequencies.get(term, 0) + 0.5))
        for term in terms
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return EvidenceAssessment(coverage=1.0, query_terms=terms)
    covered = sum(weight for term, weight in weights.items() if term in present)
    return EvidenceAssessment(
        coverage=covered / total_weight,
        query_terms=terms,
        missing_terms=[term for term in terms if term not in present],
    )
