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
