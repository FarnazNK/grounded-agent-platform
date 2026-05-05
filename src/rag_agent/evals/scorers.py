"""Scorers: pluggable evaluation metrics.

Design: a `Scorer` is any callable that takes (case, final_state) and returns
a `ScoreResult`. This is deliberately simple — no inheritance required, just
a protocol. New scorers are one function.

The built-ins cover the cases the JD calls out explicitly:
    - `ExactMatchScorer`     — routing / classification correctness
    - `ContainsScorer`       — answer content sanity checks
    - `RetrievalRecallScorer`— "did we even fetch the right doc?" (agent quality)
    - `LLMJudgeScorer`       — response quality / hallucination detection
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from rag_agent.evals.dataset import EvalCase
from rag_agent.llm import build_llm, invoke_with_retry
from rag_agent.schemas import AgentState


@dataclass
class ScoreResult:
    """One scorer's verdict on one case."""

    scorer_name: str
    score: float  # in [0, 1]
    passed: bool
    rationale: str = ""


class Scorer(Protocol):
    """Protocol every scorer satisfies."""

    name: str

    def __call__(self, case: EvalCase, state: AgentState) -> ScoreResult: ...


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------


class ExactMatchScorer:
    """Pass if the final answer exactly matches `expected_exact`.

    Useful for routing-style tests where the output is a known token like
    "refuse". Skipped automatically when a case doesn't set `expected_exact`.
    """

    name = "exact_match"

    def __call__(self, case: EvalCase, state: AgentState) -> ScoreResult:
        if case.expected_exact is None:
            return ScoreResult(self.name, 1.0, True, "skipped (no expected_exact)")

        # Special case: "refuse" checks route, not answer text.
        if case.should_refuse:
            passed = state.route == "refuse"
            return ScoreResult(
                self.name,
                float(passed),
                passed,
                f"route={state.route}, expected refuse",
            )

        actual = (state.final_answer or "").strip()
        passed = actual == case.expected_exact.strip()
        return ScoreResult(self.name, float(passed), passed, f"got: {actual[:80]!r}")


class ContainsScorer:
    """Pass if the final answer contains every substring in `expected_contains`.

    Case-insensitive. Scored as fraction of substrings present — partial
    credit is more informative than pass/fail during iteration.
    """

    name = "contains"
    pass_threshold = 1.0  # require all substrings

    def __call__(self, case: EvalCase, state: AgentState) -> ScoreResult:
        if not case.expected_contains:
            return ScoreResult(self.name, 1.0, True, "skipped (no expected_contains)")

        answer = (state.final_answer or "").lower()
        hits = [s for s in case.expected_contains if s.lower() in answer]
        score = len(hits) / len(case.expected_contains)
        passed = score >= self.pass_threshold
        missing = [s for s in case.expected_contains if s.lower() not in answer]
        rationale = f"hit {len(hits)}/{len(case.expected_contains)}"
        if missing:
            rationale += f"; missing: {missing}"
        return ScoreResult(self.name, score, passed, rationale)


class RetrievalRecallScorer:
    """Fraction of `expected_sources` present in the retrieved chunks.

    This isolates retrieval quality from generation quality — the bug you
    want to catch is "our answers got worse because retrieval regressed,"
    and a content-only scorer would conflate the two.
    """

    name = "retrieval_recall"
    pass_threshold = 1.0

    def __call__(self, case: EvalCase, state: AgentState) -> ScoreResult:
        if not case.expected_sources:
            return ScoreResult(self.name, 1.0, True, "skipped (no expected_sources)")

        retrieved_sources = {c.source for c in state.chunks}
        hits = [s for s in case.expected_sources if s in retrieved_sources]
        score = len(hits) / len(case.expected_sources)
        passed = score >= self.pass_threshold
        rationale = f"retrieved {sorted(retrieved_sources)}, expected {case.expected_sources}"
        return ScoreResult(self.name, score, passed, rationale)


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM = """You are an evaluator. Given a user question, a set of
criteria, and an assistant's answer, judge whether the answer satisfies the
criteria. Respond in JSON:
{
  "passed": boolean,
  "score": float between 0 and 1,
  "rationale": "one short sentence"
}
Be strict but fair. Penalize hallucinations (claims not supported by the
context) heavily."""


class LLMJudgeScorer:
    """Use an LLM to judge free-form answer quality.

    Caveats (worth stating up front in any real eval system):
    - LLM judges are noisy. Use them for direction, not for absolute numbers.
    - Use a different / stronger model than the one being evaluated when
      possible, to reduce self-preference bias.
    - Always pair with at least one deterministic scorer so you notice when
      the judge itself drifts.
    """

    name = "llm_judge"

    def __init__(self, llm=None) -> None:
        self._llm = llm or build_llm()

    def __call__(self, case: EvalCase, state: AgentState) -> ScoreResult:
        if not case.judge_criteria:
            return ScoreResult(self.name, 1.0, True, "skipped (no judge_criteria)")

        answer = state.final_answer or ""
        context = "\n".join(c.as_context_block() for c in state.chunks)
        user_msg = (
            f"Question:\n{case.query}\n\n"
            f"Criteria:\n{case.judge_criteria}\n\n"
            f"Context shown to the assistant:\n{context}\n\n"
            f"Assistant's answer:\n{answer}"
        )
        raw = invoke_with_retry(self._llm, [("system", _JUDGE_SYSTEM), ("user", user_msg)])
        try:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            payload = json.loads(cleaned)
            return ScoreResult(
                self.name,
                float(payload.get("score", 0.0)),
                bool(payload.get("passed", False)),
                str(payload.get("rationale", ""))[:200],
            )
        except (json.JSONDecodeError, ValueError):
            # Never let a flaky judge pass silently — mark as failed.
            return ScoreResult(self.name, 0.0, False, f"judge parse failed: {raw[:120]}")
