"""Multi-agent developer workflow: planner -> coder -> reviewer.

Each role is the existing policy-controlled :class:`DeveloperAgentHarness` with its own
tool allowlist, so role boundaries are enforced by the runtime, not by the prompt:

* **planner** - read-only; inspects the repository and returns an implementation plan.
* **coder** - the only role that may write files (and only with ``allow_writes``); runs checks.
* **reviewer** - read-only plus verification commands; returns ``approve`` or
  ``request_changes``. Requested changes go back to the coder, up to a bounded number of
  review rounds.

The whole run is one trace in the tracing backend, with a nested span per agent and tool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

from rag_agent.agent.harness import DeveloperAgentHarness
from rag_agent.agent.models import (
    AgentRunResult,
    AgentTool,
    RoleRun,
    TeamRole,
    TeamRunResult,
)
from rag_agent.agent.planner import AgentPlanner
from rag_agent.agent.policy import ToolPolicy
from rag_agent.agent.skills import AgentSkill
from rag_agent.observability import get_logger
from rag_agent.tracing import get_tracer

logger = get_logger(__name__)

READ_TOOLS = [AgentTool.LIST_FILES, AgentTool.READ_FILE, AgentTool.SEARCH_CODE]

PLANNER_SKILL = AgentSkill(
    name="team-planner",
    description="Plan a change without modifying the repository.",
    instructions=(
        "You are the PLANNER in a planner/coder/reviewer team. Inspect only what you need, "
        "then finish. In arguments.summary give a short numbered implementation plan: the "
        "files to change, what to change in each, and which check command proves it works. "
        "Do not modify files."
    ),
    allowed_tools=READ_TOOLS,
)

CODER_SKILL = AgentSkill(
    name="team-coder",
    description="Implement an approved plan with the smallest justified change.",
    instructions=(
        "You are the CODER in a planner/coder/reviewer team. Follow the plan in the task. "
        "Make the smallest change that satisfies it, run a relevant check, then finish with "
        "arguments.summary describing what changed. If reviewer feedback is included, "
        "address every point."
    ),
    allowed_tools=[*READ_TOOLS, AgentTool.WRITE_FILE, AgentTool.RUN_CHECK],
)

REVIEWER_SKILL = AgentSkill(
    name="team-reviewer",
    description="Review the coder's change and approve it or request changes.",
    instructions=(
        "You are the REVIEWER in a planner/coder/reviewer team. Read the changed files and "
        "run checks if useful. Judge correctness, security, and whether the task is met. "
        "Finish with arguments.verdict set to exactly 'approve' or 'request_changes', and "
        "arguments.summary listing concrete required fixes (or why it is acceptable). "
        "Do not modify files."
    ),
    allowed_tools=[*READ_TOOLS, AgentTool.RUN_CHECK],
)

Verdict = Literal["approve", "request_changes"]


def _review_verdict(result: AgentRunResult) -> Verdict:
    """Read the reviewer's verdict; anything missing or unrecognized is not an approval."""
    for step in reversed(result.steps):
        if step.decision.tool == AgentTool.FINISH:
            raw = str(step.decision.arguments.get("verdict", "")).strip().lower()
            return "approve" if raw == "approve" else "request_changes"
    return "request_changes"


class DeveloperAgentTeam:
    """Orchestrates planner, coder and reviewer agents over one repository."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        planner: AgentPlanner,
        role_planners: Mapping[TeamRole, AgentPlanner] | None = None,
        policy: ToolPolicy | None = None,
        max_steps_per_agent: int = 8,
        max_total_steps: int = 40,
        max_review_rounds: int = 2,
        max_context_chars: int = 40_000,
    ) -> None:
        policy = policy or ToolPolicy()
        read_only = replace(policy, allow_writes=False)
        planners = dict(role_planners or {})
        self.max_total_steps = max_total_steps
        self.max_review_rounds = max_review_rounds
        self.max_steps_per_agent = max_steps_per_agent

        def harness(role: TeamRole, role_policy: ToolPolicy) -> DeveloperAgentHarness:
            return DeveloperAgentHarness(
                repo_root,
                planner=planners.get(role, planner),
                policy=role_policy,
                max_steps=max_steps_per_agent,
                max_context_chars=max_context_chars,
            )

        # Planner and reviewer can never write, whatever the caller's policy says.
        self.agents = {
            TeamRole.PLANNER: harness(TeamRole.PLANNER, read_only),
            TeamRole.CODER: harness(TeamRole.CODER, policy),
            TeamRole.REVIEWER: harness(TeamRole.REVIEWER, read_only),
        }
        self.skills = {
            TeamRole.PLANNER: PLANNER_SKILL,
            TeamRole.CODER: CODER_SKILL,
            TeamRole.REVIEWER: REVIEWER_SKILL,
        }

    def run(self, task: str) -> TeamRunResult:
        tracer = get_tracer()
        run_id = str(uuid4())
        with (
            tracer.trace_attributes(
                trace_name="dev-team",
                session_id=run_id,
                tags=["multi-agent", "dev-team"],
            ),
            tracer.observe("dev-team", kind="agent", input={"task": task}) as span,
        ):
            result = self._run(run_id, task)
            span.update(
                output={"status": result.status, "summary": result.summary},
                metadata={
                    "review_rounds": result.review_rounds,
                    "total_steps": result.total_steps,
                    "changed_files": result.changed_files,
                },
                level="DEFAULT" if result.status == "completed" else "WARNING",
            )
            if result.review_verdict is not None:
                tracer.score(
                    "review_approved",
                    1.0 if result.review_verdict == "approve" else 0.0,
                    comment=result.summary[:500],
                )
            return result

    def _run_role(
        self, role: TeamRole, task: str, round_: int, state: TeamRunResult
    ) -> AgentRunResult | None:
        remaining = self.max_total_steps - state.total_steps
        if remaining <= 0:
            return None
        agent = self.agents[role]
        agent.max_steps = min(self.max_steps_per_agent, remaining)
        result = agent.run(task, skill=self.skills[role], role=role.value)
        state.runs.append(RoleRun(role=role, round=round_, result=result))
        state.total_steps += len(result.steps)
        return result

    def _run(self, run_id: str, task: str) -> TeamRunResult:
        state = TeamRunResult(run_id=run_id, task=task, status="blocked")
        logger.info("agent_team_started", team_run_id=run_id, task=task)

        plan_result = self._run_role(TeamRole.PLANNER, task, 0, state)
        if plan_result is None or plan_result.status != "completed":
            return self._finish(state, plan_result, "Planner did not produce a plan.")
        state.plan = plan_result.summary

        changed: set[str] = set()
        feedback = ""
        for round_ in range(1, self.max_review_rounds + 1):
            coder_task = f"{task}\n\nImplementation plan:\n{state.plan}"
            if feedback:
                coder_task += f"\n\nReviewer feedback to address:\n{feedback}"
            coder_result = self._run_role(TeamRole.CODER, coder_task, round_, state)
            if coder_result is None or coder_result.status != "completed":
                state.changed_files = sorted(changed)
                return self._finish(state, coder_result, "Coder did not complete.")
            changed.update(coder_result.changed_files)
            state.changed_files = sorted(changed)

            review_task = (
                f"Review the change for this task:\n{task}\n\n"
                f"Plan:\n{state.plan}\n\n"
                f"Coder summary:\n{coder_result.summary}\n\n"
                f"Changed files: {', '.join(state.changed_files) or 'none'}"
            )
            review_result = self._run_role(TeamRole.REVIEWER, review_task, round_, state)
            if review_result is None or review_result.status != "completed":
                return self._finish(state, review_result, "Reviewer did not complete.")

            state.review_rounds = round_
            state.review_verdict = _review_verdict(review_result)
            logger.info(
                "agent_team_review",
                team_run_id=run_id,
                round=round_,
                verdict=state.review_verdict,
            )
            if state.review_verdict == "approve":
                state.status = "completed"
                state.summary = f"Approved after {round_} review round(s). {coder_result.summary}"
                return state
            feedback = review_result.summary

        state.status = "changes_requested"
        state.summary = (
            f"Reviewer still requested changes after {self.max_review_rounds} round(s): {feedback}"
        )
        return state

    def _finish(
        self, state: TeamRunResult, last: AgentRunResult | None, reason: str
    ) -> TeamRunResult:
        if last is None:
            state.status = "max_steps"
            state.summary = f"Team step budget of {self.max_total_steps} exhausted. {reason}"
        else:
            state.status = "max_steps" if last.status == "max_steps" else "blocked"
            state.summary = f"{reason} {last.summary}".strip()
        logger.warning("agent_team_stopped", team_run_id=state.run_id, status=state.status)
        return state
