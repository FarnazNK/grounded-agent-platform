# ADR 0008: Policy-controlled developer-agent harness

## Status

Accepted.

## Context

Retrieval quality alone is not enough for software-engineering agents. They need
repository context, reusable task instructions, tools that can inspect and change code,
verification, and a trace of what happened. Giving a model unrestricted filesystem or
shell access would make those capabilities difficult to reason about and unsafe to
operate.

## Decision

Add a repository-scoped agent harness with:

- deterministic task-to-context selection;
- a JSON decision interface for model planning;
- a registry of narrow repository tools;
- declarative skills with tool allowlists;
- explicit write opt-in;
- verification-command allowlisting;
- structured per-step traces;
- independent context-selection evaluations in CI.

Filesystem and command tools remain local-only and are not exposed through the hosted
FastAPI surface.

## Consequences

The architecture can support code-oriented agent workflows while keeping model decisions
separate from execution policy. The initial policy is defense in depth, not a replacement
for OS-level isolation; production write-capable execution should run in ephemeral
sandboxed compute.
