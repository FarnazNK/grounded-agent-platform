"""Input guardrails — applied to the user query before the agent runs.

These are intentionally regex-based, not model-based. Three reasons:
1. Latency: they need to fire in <1 ms.
2. Predictability: the eval suite needs deterministic behavior.
3. Auditability: a regex is reviewable; an LLM judge isn't.

Model-based guardrails belong on top of these, not instead of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_agent.guardrails.base import GuardrailAction, GuardrailDecision

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

# These are conservative — they aim for high precision, not exhaustive coverage.
# Production systems should layer Microsoft Presidio or similar on top.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


@dataclass
class PIIDetector:
    """Detects PII in input. Configurable to either sanitize (replace with
    placeholders) or block.

    The default is SANITIZE — most enterprise deployments want the request to
    proceed with the PII redacted, not be hard-blocked. Set `mode="block"` for
    hard-block behavior.
    """

    name: str = "pii_detector"
    mode: str = "sanitize"  # "sanitize" or "block"
    detect_types: tuple[str, ...] = (
        "email",
        "phone_us",
        "ssn",
        "credit_card",
    )  # ip_address omitted by default — too noisy

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
            return GuardrailDecision(self.name, GuardrailAction.ALLOW, "no pii detected")

        if self.mode == "block":
            return GuardrailDecision(
                self.name,
                GuardrailAction.BLOCK,
                f"pii detected: {sorted(hits.keys())}",
                metadata={"hits": hits},
            )

        return GuardrailDecision(
            self.name,
            GuardrailAction.SANITIZE,
            f"pii redacted: {sorted(hits.keys())}",
            sanitized_text=sanitized,
            metadata={"hits": hits},
        )


# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------

# These catch the most common, low-effort injection attempts. They will not
# stop a determined attacker — that's what defense in depth is for. They will
# stop drive-by junk that pollutes logs and skews eval metrics.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\bignore\b.{0,30}\b(previous|prior|above)\b.{0,30}\b(instruction|prompt|rule)", re.I
    ),  # noqa: E501
    re.compile(
        r"\bdisregard\b.{0,30}\b(previous|prior|above)\b.{0,30}\b(instruction|prompt|rule)", re.I
    ),  # noqa: E501
    re.compile(
        r"\b(reveal|show|print|leak|disclose).{0,30}\b(system|hidden)\b.{0,30}\bprompt\b", re.I
    ),  # noqa: E501
    re.compile(r"\byou are now\b.{0,40}\b(dan|jailbreak|unrestricted|developer mode)\b", re.I),
    re.compile(
        r"\b(repeat|echo|output)\b.{0,30}\b(verbatim|exactly|word for word)\b"
        r".{0,30}\b(system|prompt|instruction)\b",
        re.I,
    ),
    re.compile(r"<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]"),  # chat template injection
]


@dataclass
class PromptInjectionDetector:
    """Detects common prompt injection patterns.

    Always blocks rather than sanitizing — there's no safe sanitization for an
    injection attempt that doesn't change the user's apparent intent. If we
    flag it, we refuse it.
    """

    name: str = "prompt_injection_detector"

    def __call__(self, text: str) -> GuardrailDecision:
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return GuardrailDecision(
                    self.name,
                    GuardrailAction.BLOCK,
                    f"prompt injection pattern matched: {match.group(0)[:60]!r}",
                    metadata={"pattern": pattern.pattern},
                )
        return GuardrailDecision(self.name, GuardrailAction.ALLOW, "no injection detected")


# ---------------------------------------------------------------------------
# Profanity / abuse — minimal, illustrative
# ---------------------------------------------------------------------------

# Deliberately tiny — full profanity lists are a maintenance burden and
# context-blind. Real systems should use Detoxify or Perspective API.
_ABUSIVE_TERMS = {
    "kill yourself",
    "shut up",
    "you're an idiot",
}


@dataclass
class ProfanityFilter:
    """Catches a small set of obviously abusive phrases. Sanitizes by default."""

    name: str = "profanity_filter"
    mode: str = "sanitize"

    def __call__(self, text: str) -> GuardrailDecision:
        lowered = text.lower()
        hits = [term for term in _ABUSIVE_TERMS if term in lowered]
        if not hits:
            return GuardrailDecision(self.name, GuardrailAction.ALLOW, "clean")

        if self.mode == "block":
            return GuardrailDecision(
                self.name,
                GuardrailAction.BLOCK,
                f"abusive language: {hits}",
                metadata={"hits": hits},
            )

        sanitized = text
        for term in hits:
            sanitized = re.sub(re.escape(term), "[redacted]", sanitized, flags=re.I)
        return GuardrailDecision(
            self.name,
            GuardrailAction.SANITIZE,
            f"abusive language redacted: {hits}",
            sanitized_text=sanitized,
            metadata={"hits": hits},
        )
