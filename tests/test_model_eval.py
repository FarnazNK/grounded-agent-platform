"""Judge, scoring and real-model eval plumbing, using fake providers (no API calls)."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from rag_agent.config import Settings
from rag_agent.evals import runner as eval_runner
from rag_agent.evals.dataset import EvalCase
from rag_agent.evals.judge import LLMJudge, parse_verdict
from rag_agent.evals.runner import TokenPrices, run_evaluation
from rag_agent.evals.scorers import score_case, should_abstain
from rag_agent.llm import LLMResult, _token_counts
from rag_agent.schemas import QueryResult, RetrievedChunk, UsageInfo


def _usage(prompt: int = 100, completion: int = 20) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


@pytest.mark.parametrize(
    ("text", "passed"),
    [
        ('{"verdict": "pass", "reason": "ok"}', True),
        ('Here you go: {"verdict": "FAIL", "reason": "invented a number"}', False),
        ("pass", False),  # not JSON: counts as a fail
        ('{"verdict": "maybe"}', False),
    ],
)
def test_parse_verdict(text, passed):
    assert parse_verdict(text)[0] is passed


class FakeJudgeProvider:
    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = verdicts
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> LLMResult:
        self.prompts.append(user)
        verdict = self.verdicts[len(self.prompts) - 1]
        return LLMResult(
            text=f'{{"verdict": "{verdict}", "reason": "because"}}',
            usage=_usage(50, 10),
            latency_ms=1.0,
        )


def test_judge_sends_question_criteria_and_answer():
    provider = FakeJudgeProvider(["pass"])
    verdict = LLMJudge(provider).grade(
        question="How many PTO days?", criteria="Says 20 days.", answer="20 days."
    )
    assert verdict.passed
    assert verdict.prompt_tokens == 50
    prompt = provider.prompts[0]
    assert "How many PTO days?" in prompt and "Says 20 days." in prompt and "20 days." in prompt


def _result(answer: str, citations: list[str], refusal: bool = False) -> QueryResult:
    return QueryResult(
        answer=answer,
        citations=citations,
        chunks=[
            RetrievedChunk(
                chunk_id="c",
                document_id="d",
                source_name="pto_policy.md",
                content="x",
                fused_score=1,
            )
        ],
        refusal=refusal,
        grounded=True,
        usage=_usage(),
    )


def test_abstention_is_scored_only_for_unanswerable_cases():
    unanswerable = EvalCase(case_id="x", query="weather?", tags=["out_of_scope"])
    injection = EvalCase(case_id="y", query="ignore rules", tags=["prompt_injection"])
    answerable = EvalCase(case_id="z", query="pto?", expected_sources=["pto_policy.md"])
    assert should_abstain(unanswerable)
    assert not should_abstain(injection)
    assert not should_abstain(answerable)

    declined = score_case(unanswerable, _result("The documents don't cover weather.", []))
    answered = score_case(
        unanswerable, _result("It will be sunny. [source: pto_policy.md]", ["pto_policy.md"])
    )
    assert declined.abstained is True
    assert answered.abstained is False
    assert score_case(answerable, _result("20 days", ["pto_policy.md"])).abstained is None


def test_token_counts_prefer_provider_usage():
    anthropic_style = SimpleNamespace(usage_metadata={"input_tokens": 812, "output_tokens": 64})
    openai_style = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 500, "completion_tokens": 40}},
    )
    bare = SimpleNamespace(usage_metadata=None, response_metadata={})
    assert _token_counts(anthropic_style, "q", "a") == (812, 64)
    assert _token_counts(openai_style, "q", "a") == (500, 40)
    assert _token_counts(bare, "one two", "three") == (2, 1)


def test_token_prices():
    assert TokenPrices(3.0, 15.0).cost(1_000_000, 100_000) == pytest.approx(4.5)


class FakeModel:
    """Stands in for a real LLM: answers from the top chunk and cites it,
    declines when nothing relevant was retrieved."""

    def answer_question(self, query: str, chunks: list[RetrievedChunk]) -> LLMResult:
        if not chunks:
            text = "I could not find that in the documents."
        else:
            text = f"{chunks[0].content[:200]} [source: {chunks[0].source_name}]"
        return LLMResult(text=text, usage=_usage(300, 40), latency_ms=5.0)

    def complete(self, system: str, user: str) -> LLMResult:
        return LLMResult(text='{"ranking": []}', usage=_usage(80, 5), latency_ms=1.0)


def test_real_model_path_with_judge_and_costs(monkeypatch):
    monkeypatch.setattr(eval_runner, "build_llm_provider", lambda settings: FakeModel())
    judge = LLMJudge(FakeJudgeProvider(["pass", "fail", "pass", "pass"] * 20))
    report = run_evaluation(
        "data/eval_datasets/hr_smoke.yaml",
        Settings(app_env="test"),
        use_configured_llm=True,
        judge=judge,
        prices=TokenPrices(input_per_mtok=1.0, output_per_mtok=2.0),
    )
    metrics = report.metrics
    assert metrics["judged_cases"] == metrics["cases"]
    assert 0.0 < metrics["judge_pass_rate"] < 1.0
    assert metrics["estimated_cost_usd_total"] > 0
    assert report.config["judge"] == "on"
    assert all(case.judge is not None for case in report.cases)


def test_offline_comparison_script_writes_report(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_model_eval.py",
            "--offline",
            "--datasets",
            "data/eval_datasets/quality.yaml",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = (tmp_path / "report.md").read_text()
    assert "| Metric | reranker: none | reranker: lexical |" in report
    assert (tmp_path / "results.json").exists()
    assert "Citation correctness" in completed.stdout


def test_real_model_run_requires_a_configured_model(tmp_path):
    completed = subprocess.run(
        [sys.executable, "scripts/run_model_eval.py", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "APP_LLM_PROVIDER": "deterministic"},
    )
    assert completed.returncode != 0
    assert "APP_LLM_PROVIDER" in completed.stderr
