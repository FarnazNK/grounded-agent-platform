"""Security audit events: who did what, to which workspace resource, with what outcome.

Events are emitted as one JSON object per line on the ``rag_agent.audit`` logger, so any log
shipper (Vector, Fluent Bit, CloudWatch, Loki) can route them to a SIEM or WORM storage
without extra infrastructure. Each line carries ``"audit": true`` for filtering.

Privacy rules, enforced here rather than left to call sites:
* Query text and email addresses are never logged; a salted-free SHA-256 prefix is logged
  instead, which lets investigators correlate repeated events without storing the content.
* Details are restricted to scalar values and short lists so documents cannot leak in.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from rag_agent.observability import get_logger

_log = get_logger("rag_agent.audit")

Outcome = Literal["success", "denied", "failure"]
_MAX_LIST_ITEMS = 20
_MAX_STR_LEN = 200


def fingerprint(value: str) -> str:
    """Stable, non-reversible identifier for sensitive strings (queries, emails)."""
    return hashlib.sha256(value.strip().casefold().encode("utf-8")).hexdigest()[:16]


def _clean(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:_MAX_STR_LEN]
    if isinstance(value, list | tuple | set | frozenset):
        return [_clean(item) for item in list(value)[:_MAX_LIST_ITEMS]]
    return str(value)[:_MAX_STR_LEN]


def audit_event(
    action: str,
    *,
    outcome: Outcome,
    actor_id: str | None = None,
    workspace_id: str | None = None,
    resource_id: str | None = None,
    client_ip: str | None = None,
    **details: Any,
) -> None:
    _log.info(
        "audit",
        audit=True,
        action=action,
        outcome=outcome,
        actor_id=actor_id,
        workspace_id=workspace_id,
        resource_id=resource_id,
        client_ip=client_ip,
        details={key: _clean(value) for key, value in details.items()},
    )
