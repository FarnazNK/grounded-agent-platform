# Developer Agent Harness

The developer-agent surface extends the retrieval system with a repository-aware,
policy-controlled execution loop for software engineering tasks.

## Goals

- Give an LLM compact, task-relevant repository context instead of dumping an entire codebase.
- Keep tool execution explicit, inspectable, and independently policy-enforced.
- Make reusable behavior declarative through skills.
- Record every decision and tool result as a run trace.
- Evaluate context selection separately from model quality.
- Keep filesystem and command execution local by default rather than exposing them through the public API.

## Flow

```text
developer task
    |
    v
repository context builder
    |
    v
optional reusable skill
    |
    v
LLM tool planner
    |
    v
tool registry ---------> list/read/search
    |                   write (explicit opt-in)
    |                   verification commands
    v
execution policy
    |
    v
repository workspace
    |
    v
trace + changed files + verification output
```

The model does not get direct shell or filesystem access. It chooses from registered
tools and the runtime validates the request again before execution.

## Context engineering

`RepositoryContextBuilder` ranks text/code files using task tokens found in paths and
file contents, applies file-size limits, ignores generated/vendor directories, and
enforces a total context budget. This is intentionally deterministic so context quality
can be measured independently from the LLM.

The CI evaluation in `scripts/run_agent_evals.py` measures whether known engineering
tasks retrieve at least one expected implementation file in the top-k context.

## Tool policy

`ToolPolicy` provides defense in depth:

- repository-root confinement prevents path traversal;
- credential-like paths and files are blocked;
- writes are disabled unless explicitly enabled for the run;
- write size is bounded;
- commands run without a shell;
- only verification-oriented command prefixes are allowed;
- subprocesses receive a small environment allowlist rather than model-provider secrets;
- command execution has a hard timeout.

This is a portfolio harness, not an OS-level sandbox. A production service should run
write-enabled agents in ephemeral containers or VMs with filesystem, network, CPU,
memory, and credential isolation.

## Skills

Reusable behavior lives in `agent_skills/*.yaml`. A skill supplies instructions and a
tool allowlist without changing the harness. Included examples cover code changes,
debugging, and read-only review.

## Local usage

The CLI uses the configured model provider. Deterministic mode performs a context-only
dry run. Anthropic or OpenAI mode can iterate through repository tools.

```bash
# Read-only context/tool run
rag-agent dev-agent --repo . --task "Trace request authentication"

# Explicitly permit file edits for a local coding task
APP_LLM_PROVIDER=anthropic \
APP_LLM_MODEL=<model-name> \
rag-agent dev-agent \
  --repo . \
  --skill code-change \
  --allow-writes \
  --task "Add validation and tests for the new field"
```

Do not enable write-capable repository tools on an internet-facing API without a
separate sandbox boundary and stronger authorization.

## Multi-agent team

`DeveloperAgentTeam` (`src/rag_agent/agent/team.py`, CLI `rag-agent dev-team`) composes three
harness runs instead of one long loop:

```text
task
  |
  v
planner  (read-only policy) ---------> numbered plan
  |
  v
coder    (caller's policy; writes only with --allow-writes) --> changed files
  |
  v
reviewer (read-only policy + checks) --> approve | request_changes
  |                                          |
  | approve                                  | feedback (bounded rounds)
  v                                          v
completed                                  back to coder
```

Design choices:

- **Role boundaries are enforced by the runtime.** Each role is an `AgentSkill` with its own
  tool allowlist, and planner/reviewer harnesses are built with `allow_writes=False`, so a
  prompt-injected reviewer still cannot write files.
- **Fail closed on review.** A missing or unrecognized verdict counts as
  `request_changes`.
- **Bounded cost.** Per-agent `max_steps`, a shared `max_total_steps` budget, and
  `max_review_rounds`; exhausting them returns `max_steps` or `changes_requested` rather
  than looping.
- **Per-role models.** `role_planners` lets each role use a different planner, for example
  a cheaper model for review.
- **One trace per team run** when Langfuse tracing is enabled (see `docs/observability.md`).

Results are returned as `TeamRunResult`: status, plan, verdict, review rounds, total steps,
changed files, and every role's full `AgentRunResult`.
