"""Output guardrails — applied to the agent's final answer before returning.

The risks are different from input:
- The model might leak PII it pulled from a context chunk.
- The model might generate an answer without citing a source.
- The model might hallucinate sources that weren't in the context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_agent.guardrails.base import GuardrailAction, GuardrailDecision
from rag_agent.guardrails.input_filters import _PII_PATTERNS  # reuse the same patterns


@dataclass
class PIILeakDetector:
    """Detects PII in the model's output. Sanitizes by replacing with placeholders.

    Different default than the input filter: outputs are *always* sanitized
    rather than blocked — refusing after we already paid for the generation
    is wasteful, and the user is likely better served by a redacted answer
    than no answer at all.
    """

    name: str = "pii_leak_detector"
    detect_types: tuple[str, ...] = ("email", "phone_us", "ssn", "credit_card")

    def __call__(self, text: str) -> GuardrailDecision:
        hits: dict[str, list[str]] = {}
        sanitized = text
        for kind in self.detect_types:
            pattern = _PII_PATTERNS.get(kind)
            if pattern is None:
                continue
            matches = pattern.findall(text)
            if matches:
                hits[kind] = matches
                sanitized = pattern.sub(f"[REDACTED_{kind.upper()}]", sanitized)

        if not hits:
            return GuardrailDecision(self.name, GuardrailAction.ALLOW, "no pii in output")

        return GuardrailDecision(
            self.name,
            GuardrailAction.SANITIZE,
            f"output pii redacted: {sorted(hits.keys())}",
            sanitized_text=sanitized,
            metadata={"hits": hits},
        )


@dataclass
class SourceCitationChecker:
    """Verifies that any [source: ...] citations point to chunks actually
    retrieved this turn. Catches the common hallucination where the model
    invents plausible-looking source names.

    Designed to be wired in by the API layer, not by the bare guardrail
    pipeline (it needs context the pipeline doesn't have). Exposed here for
    completeness; instantiate with the list of valid source names per request.
    """

    valid_sources: frozenset[str] = frozenset()
    name: str = "source_citation_checker"

    _CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.I)

    def __call__(self, text: str) -> GuardrailDecision:
        cited = {m.group(1).strip() for m in self._CITATION_RE.finditer(text)}
        if not cited:
            return GuardrailDecision(self.name, GuardrailAction.ALLOW, "no citations to verify")
        invalid = cited - self.valid_sources
        if invalid:
            return GuardrailDecision(
                self.name,
                GuardrailAction.SANITIZE,
                f"invalid source citations: {sorted(invalid)}",
                # Strip the bad citations rather than blocking — the surrounding
                # answer may still be useful.
                sanitized_text=self._strip_invalid(text, invalid),
                metadata={"invalid_sources": sorted(invalid), "cited": sorted(cited)},
            )
        return GuardrailDecision(self.name, GuardrailAction.ALLOW, "all citations valid")

    def _strip_invalid(self, text: str, invalid: set[str]) -> str:
        def _maybe_strip(match: re.Match[str]) -> str:
            src = match.group(1).strip()
            return "[source: unverified]" if src in invalid else match.group(0)

        return self._CITATION_RE.sub(_maybe_strip, text)
