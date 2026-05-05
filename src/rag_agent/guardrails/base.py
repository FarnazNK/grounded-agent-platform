"""Guardrail base types.

Design mirrors the eval scorers: a `Guardrail` is anything callable with the
right signature. No inheritance, no framework. Decisions are explicit so the
caller can choose to block, sanitize, or just log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class GuardrailAction(StrEnum):
    """What the guardrail wants the caller to do."""

    ALLOW = "allow"  # safe; pass through
    SANITIZE = "sanitize"  # replace `sanitized_text` and continue
    BLOCK = "block"  # refuse the request


@dataclass
class GuardrailDecision:
    """A single guardrail's verdict on a single piece of text."""

    guardrail_name: str
    action: GuardrailAction
    reason: str = ""
    sanitized_text: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.action == GuardrailAction.ALLOW


class Guardrail(Protocol):
    """Protocol every guardrail satisfies."""

    name: str

    def __call__(self, text: str) -> GuardrailDecision: ...


@dataclass
class GuardrailViolation(Exception):
    """Raised when a BLOCK action fires and the caller wants to short-circuit."""

    decision: GuardrailDecision

    def __str__(self) -> str:
        return f"[{self.decision.guardrail_name}] {self.decision.reason}"


def apply_guardrails(
    text: str,
    guardrails: list[Guardrail],
    *,
    raise_on_block: bool = True,
) -> tuple[str, list[GuardrailDecision]]:
    """Run every guardrail in order. Returns the (possibly sanitized) text and
    the list of decisions for logging.

    If any guardrail returns BLOCK and `raise_on_block` is True (the default),
    raises `GuardrailViolation`. Otherwise the caller can inspect the decisions
    list and decide what to do.

    Sanitization is cumulative: each guardrail sees the output of the previous one.
    """
    decisions: list[GuardrailDecision] = []
    current = text
    for g in guardrails:
        decision = g(current)
        decisions.append(decision)
        if decision.action == GuardrailAction.BLOCK:
            if raise_on_block:
                raise GuardrailViolation(decision)
            return current, decisions
        if decision.action == GuardrailAction.SANITIZE and decision.sanitized_text is not None:
            current = decision.sanitized_text
    return current, decisions
