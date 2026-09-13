"""Guardrails — input and output filtering before/after the agent runs.

The principle here is the same as eval scoring: every guardrail is a callable
matching a Protocol. Adding a new check is a function, not a subclass.
Guardrails are applied in order; the first one that flags blocks the request.
"""

from __future__ import annotations

from rag_agent.guardrails.base import (
    Guardrail,
    GuardrailDecision,
    GuardrailViolation,
    apply_guardrails,
)
from rag_agent.guardrails.input_filters import (
    DataExfiltrationDetector,
    PIIDetector,
    ProfanityFilter,
    PromptInjectionDetector,
)
from rag_agent.guardrails.output_filters import (
    PIILeakDetector,
    SourceCitationChecker,
)

__all__ = [
    "Guardrail",
    "GuardrailDecision",
    "GuardrailViolation",
    "DataExfiltrationDetector",
    "PIIDetector",
    "PIILeakDetector",
    "ProfanityFilter",
    "PromptInjectionDetector",
    "SourceCitationChecker",
    "apply_guardrails",
]
