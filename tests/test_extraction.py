from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

from rag_agent.extraction import (
    DeterministicExtractor,
    extract,
    validate_output,
)
from tests.conftest import auth_header

DOC = """# Paid Time Off Policy

All full-time employees accrue paid time off monthly.

## Carryover
Employees may carry over up to 5 unused PTO days into the next calendar year.

## Requesting time off
Submit requests at least 2 weeks in advance. Manager approval is required.
"""

GOOD = {
    "policy_name": {"value": "Paid Time Off Policy", "quote": "# Paid Time Off Policy"},
    "applies_to": [
        {
            "value": "full-time employees",
            "quote": "All full-time employees accrue paid time off monthly.",
        }
    ],
    "numeric_limits": [
        {
            "label": "Maximum PTO carryover",
            "value": 5,
            "unit": "days",
            "quote": (
                "Employees may carry over up to 5 unused PTO days into the next calendar year."
            ),
        },
        {
            "label": "Advance notice for requests",
            "value": 2,
            "unit": "weeks",
            "quote": "Submit requests at least 2 weeks in advance.",
        },
    ],
    "approval_required": {"value": True, "quote": "Manager approval is required."},
}


def _with(**changes):
    data = json.loads(json.dumps(GOOD))
    for path, value in changes.items():
        target = data
        *parents, last = path.split("__")
        for p in parents:
            target = target[int(p)] if p.isdigit() else target[p]
        if value is ...:
            del target[last]
        else:
            target[int(last) if last.isdigit() else last] = value
    return json.dumps(data)


# ---------------------------------------------------------------- validator


def test_grounded_output_passes():
    result = validate_output(json.dumps(GOOD), DOC)
    assert result.ok, result.errors
    assert result.record is not None
    assert result.record.numeric_limits[0].value == 5


def test_fenced_json_is_accepted():
    assert validate_output(f"```json\n{json.dumps(GOOD)}\n```", DOC).ok


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sure! Here are the facts: ...", "not valid JSON"),
        ("[1, 2]", "single JSON object"),
        (_with(numeric_limits__0__unit="fortnights"), "numeric_limits.0.unit"),
        (_with(numeric_limits__0__value=-3), "numeric_limits.0.value"),
        (_with(policy_name__confidence=0.9), "policy_name.confidence: Extra inputs"),
        (_with(policy_name=...), "policy_name: Field required"),
        (_with(approval_required__value="maybe"), "approval_required.value"),
    ],
)
def test_schema_violations_are_rejected(raw, expected):
    result = validate_output(raw, DOC)
    assert not result.ok
    assert any(expected in e for e in result.errors), result.errors


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Invented quote that isn't in the document.
        (
            _with(numeric_limits__0__quote="Employees may carry over up to 10 days."),
            "numeric_limits.0.quote: not found",
        ),
        # Real quote, wrong number.
        (_with(numeric_limits__0__value=10), "numeric_limits.0.value: 10 does not appear"),
        # Value not supported by its quote.
        (_with(applies_to__0__value="contractors"), "applies_to.0.value"),
        (
            _with(approval_required__quote="Director approval is always required."),
            "approval_required.quote: not found",
        ),
    ],
)
def test_ungrounded_output_is_rejected(raw, expected):
    result = validate_output(raw, DOC)
    assert not result.ok
    assert any(expected in e for e in result.errors), result.errors


def test_quotes_match_across_whitespace_and_case():
    raw = _with(numeric_limits__1__quote="submit REQUESTS at least\n 2 weeks   in advance.")
    assert validate_output(raw, DOC).ok


# ---------------------------------------------------------------- retry loop


class ScriptedLLM:
    name = "scripted"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self.replies.pop(0)


def test_retry_with_errors_then_valid():
    llm = ScriptedLLM(_with(numeric_limits__0__value=10), json.dumps(GOOD))
    outcome = extract(DOC, llm)
    assert outcome.status == "valid"
    assert outcome.attempts == 2
    assert "10 does not appear" in llm.prompts[1]
    assert outcome.history and "numeric_limits.0.value" in outcome.history[0][0]


def test_rejected_after_two_bad_replies():
    llm = ScriptedLLM("not json", _with(applies_to__0__value="everyone"))
    outcome = extract(DOC, llm)
    assert outcome.status == "rejected"
    assert outcome.data is None
    assert outcome.attempts == 2


def test_unknown_schema():
    with pytest.raises(ValueError):
        extract(DOC, ScriptedLLM(), schema="invoices")


# ---------------------------------------------------------------- corpus gate

CORPUS = sorted(
    p for d in ("data/sample_corpus", "data/sample_corpus_extended") for p in Path(d).glob("*.md")
)


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_offline_extractor_is_valid_on_every_sample_document(path: Path):
    outcome = extract(path.read_text(), DeterministicExtractor())
    assert outcome.status == "valid", outcome.errors
    assert outcome.attempts == 1


def test_offline_extractor_reads_the_pto_policy():
    data = DeterministicExtractor.extract(Path("data/sample_corpus/pto_policy.md").read_text())
    assert data["policy_name"]["value"] == "Paid Time Off Policy"
    limits = {(lim["value"], lim["unit"]) for lim in data["numeric_limits"]}
    assert {(15.0, "days"), (5.0, "days"), (2.0, "weeks"), (10.0, "days")} <= limits
    assert (2.0, "years") not in limits  # "0-2 years of tenure" is a range, not a limit
    assert data["approval_required"]["value"] is True


# ---------------------------------------------------------------- API


def _upload(client, seeded_app, name: str, body: str) -> str:
    response = client.post(
        "/v1/documents/upload",
        params={"workspace_id": seeded_app.alice_workspace_id},
        headers=auth_header(seeded_app.alice_id, seeded_app.service.settings),
        files={"file": (name, BytesIO(body.encode()), "text/markdown")},
    )
    assert response.status_code == 200
    return response.json()["document"]["id"]


def test_api_extracts_and_saves_only_validated_output(client, seeded_app):
    document_id = _upload(client, seeded_app, "pto.md", DOC)
    headers = auth_header(seeded_app.alice_id, seeded_app.service.settings)

    response = client.post(
        f"/v1/documents/{document_id}/extract",
        headers=headers,
        json={"workspace_id": seeded_app.alice_workspace_id, "schema": "policy_facts"},
    )
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["status"] == "valid"
    assert body["schema"] == "policy_facts"
    assert body["data"]["policy_name"]["value"] == "Paid Time Off Policy"

    stored = seeded_app.service.store.get_document(document_id, seeded_app.alice_workspace_id)
    assert stored is not None
    saved = stored.metadata["extractions"]["policy_facts"]
    assert saved["data"] == body["data"]
    assert saved["provider"] == "deterministic-rules"


def test_api_rejects_invalid_output_and_saves_nothing(client, seeded_app):
    document_id = _upload(client, seeded_app, "pto2.md", DOC + "\nVersion 2.")
    seeded_app.service.structured_llm = ScriptedLLM("{}", _with(numeric_limits__0__value=99))
    try:
        response = client.post(
            f"/v1/documents/{document_id}/extract",
            headers=auth_header(seeded_app.alice_id, seeded_app.service.settings),
            json={"workspace_id": seeded_app.alice_workspace_id},
        )
    finally:
        seeded_app.service.structured_llm = DeterministicExtractor()
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "extraction_rejected"
    assert body["details"]["attempts"] == 2
    assert any("99 does not appear" in e for e in body["details"]["errors"])

    stored = seeded_app.service.store.get_document(document_id, seeded_app.alice_workspace_id)
    assert stored is not None and "extractions" not in stored.metadata


def test_api_extraction_is_tenant_scoped(client, seeded_app):
    document_id = _upload(client, seeded_app, "pto3.md", DOC + "\nVersion 3.")
    response = client.post(
        f"/v1/documents/{document_id}/extract",
        headers=auth_header(seeded_app.bob_id, seeded_app.service.settings),
        json={"workspace_id": seeded_app.alice_workspace_id},
    )
    assert response.status_code == 403
