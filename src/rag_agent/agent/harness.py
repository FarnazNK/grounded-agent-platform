"""Policy-controlled developer agent, orchestrated as a LangGraph state machine.

Each run is a small graph:

    START -> plan -> guard -> act -> plan -> ... -> END

* ``plan``  asks the planner (LLM or deterministic) for exactly one next action.
* ``guard`` enforces the per-run tool allowlist and handles ``finish``.
* ``act``   executes the tool through the policy-checked toolbox.

Every node can end the run by setting ``status``. The routing functions only
read that field, so each stop condition (planner failure, disallowed tool,
policy violation, finish, step budget) is a single, testable transition rather
than an early ``return`` buried in a loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from rag_agent.agent.context import RepositoryContextBuilder
from rag_agent.agent.models import (
    AgentDecision,
    AgentRunResult,
    AgentStep,
    AgentTool,
    ContextFile,
)
from rag_agent.agent.planner import AgentPlanner
from rag_agent.agent.policy import ToolPolicy
from rag_agent.agent.skills import AgentSkill, SkillRegistry
from rag_agent.agent.tools import RepositoryToolbox
from rag_agent.observability import get_logger, request_context

logger = get_logger(__name__)

RunStatus = Literal["completed", "blocked", "max_steps"]


class AgentState(TypedDict):
    """State carried between graph nodes for one agent run."""

    task: str
    context: list[ContextFile]
    allowed_tools: list[AgentTool]
    skill: AgentSkill | None
    history: list[AgentStep]
    changed_files: list[str]
    step: int
    decision: AgentDecision | None
    status: RunStatus | None
    summary: str


def _route(state: AgentState, next_node: str) -> str:
    return END if state["status"] is not None else next_node


class DeveloperAgentHarness:
    """Policy-controlled agent loop for repository-scoped software tasks."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        planner: AgentPlanner,
        policy: ToolPolicy | None = None,
        max_steps: int = 8,
        max_context_chars: int = 40_000,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.policy = policy or ToolPolicy()
        self.max_steps = max_steps
        self.context_builder = RepositoryContextBuilder(
            self.repo_root,
            max_context_chars=max_context_chars,
        )
        self.toolbox = RepositoryToolbox(
            self.repo_root,
            policy=self.policy,
            context_builder=self.context_builder,
        )
        self.registry = self.toolbox.registry()
        self.planner = planner
        self.skills = skill_registry or SkillRegistry.from_directory(
            self.repo_root / "agent_skills"
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ graph

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("guard", self._guard)
        graph.add_node("act", self._act)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", lambda s: _route(s, "guard"), ["guard", END])
        graph.add_conditional_edges("guard", lambda s: _route(s, "act"), ["act", END])
        graph.add_conditional_edges("act", lambda s: _route(s, "plan"), ["plan", END])
        return graph.compile()

    def _plan(self, state: AgentState) -> dict[str, Any]:
        try:
            decision = self.planner.decide(
                task=state["task"],
                context=state["context"],
                history=state["history"],
                allowed_tools=state["allowed_tools"],
                skill=state["skill"],
            )
        except Exception as exc:
            logger.error("agent_planner_failed", error=str(exc), step=state["step"])
            return {"status": "blocked", "summary": f"Planner failed: {exc}"}
        return {"decision": decision}

    def _guard(self, state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        assert decision is not None  # set by _plan before routing here
        index = state["step"]

        if decision.tool not in state["allowed_tools"]:
            logger.warning(
                "agent_tool_blocked",
                step=index,
                tool=decision.tool.value,
                reason="tool_not_allowed_for_run",
            )
            return {
                "history": [*state["history"], AgentStep(index=index, decision=decision)],
                "status": "blocked",
                "summary": f"Tool '{decision.tool.value}' is not allowed for this run.",
            }

        if decision.tool == AgentTool.FINISH:
            logger.info(
                "agent_run_completed",
                step=index,
                changed_files=state["changed_files"],
            )
            return {
                "history": [*state["history"], AgentStep(index=index, decision=decision)],
                "status": "completed",
                "summary": str(
                    decision.arguments.get("summary") or decision.reason or "Agent run completed."
                ),
            }

        return {}

    def _act(self, state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        assert decision is not None
        index = state["step"]

        result = self.registry.execute(decision.tool, decision.arguments)
        logger.info(
            "agent_tool_executed",
            step=index,
            tool=decision.tool.value,
            success=result.success,
            metadata=result.metadata,
        )
        step = AgentStep(index=index, decision=decision, result=result)
        update: dict[str, Any] = {"history": [*state["history"], step], "step": index + 1}

        changed_path = result.metadata.get("path")
        wrote = decision.tool == AgentTool.WRITE_FILE and result.success
        if wrote and isinstance(changed_path, str):
            update["changed_files"] = sorted({*state["changed_files"], changed_path})

        if result.metadata.get("blocked"):
            update.update(status="blocked", summary=result.output)
        elif index + 1 >= self.max_steps:
            logger.warning("agent_run_max_steps", max_steps=self.max_steps)
            update.update(
                status="max_steps",
                summary=f"Agent reached the maximum of {self.max_steps} steps.",
            )
        return update

    # -------------------------------------------------------------------- run

    def run(self, task: str, *, skill_name: str | None = None) -> AgentRunResult:
        run_id = str(uuid4())
        context = self.context_builder.build(task)
        skill = self.skills.get(skill_name)

        if skill_name and skill is None:
            logger.warning("agent_skill_not_found", run_id=run_id, skill=skill_name)
            return AgentRunResult(
                run_id=run_id,
                task=task,
                status="blocked",
                selected_context=context,
                summary=f"Unknown agent skill: {skill_name}",
                skill=skill_name,
            )

        allowed = set(self.registry.names)
        if not self.policy.allow_writes:
            allowed.discard(AgentTool.WRITE_FILE)
        if skill is not None and skill.allowed_tools:
            allowed &= set(skill.allowed_tools)
        allowed_tools = sorted(allowed | {AgentTool.FINISH}, key=str)

        logger.info(
            "agent_run_started",
            run_id=run_id,
            task=task,
            skill=skill.name if skill else None,
            context_files=[item.path for item in context],
            allowed_tools=[tool.value for tool in allowed_tools],
        )

        initial: AgentState = {
            "task": task,
            "context": context,
            "allowed_tools": allowed_tools,
            "skill": skill,
            "history": [],
            "changed_files": [],
            "step": 0,
            "decision": None,
            "status": None,
            "summary": "",
        }
        if self.max_steps <= 0:
            initial["status"] = "max_steps"
            initial["summary"] = f"Agent reached the maximum of {self.max_steps} steps."
            final = initial
        else:
            with request_context(agent_run_id=run_id):
                # Each step visits at most three nodes; the budget is enforced in `act`,
                # so this limit is only a backstop against a routing bug.
                final = self.graph.invoke(
                    initial,
                    config={"recursion_limit": 3 * self.max_steps + 3},
                )

        status = final["status"] or "max_steps"
        return AgentRunResult(
            run_id=run_id,
            task=task,
            status=status,
            selected_context=context,
            steps=final["history"],
            summary=final["summary"],
            changed_files=final["changed_files"],
            skill=skill.name if skill else None,
        )
