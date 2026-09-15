from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rag_agent.agent.context import RepositoryContextBuilder
from rag_agent.agent.models import AgentRunResult, AgentStep, AgentTool
from rag_agent.agent.planner import AgentPlanner
from rag_agent.agent.policy import ToolPolicy
from rag_agent.agent.skills import SkillRegistry
from rag_agent.agent.tools import RepositoryToolbox
from rag_agent.observability import get_logger, request_context

logger = get_logger(__name__)


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

    def run(self, task: str, *, skill_name: str | None = None) -> AgentRunResult:
        run_id = str(uuid4())
        context = self.context_builder.build(task)
        history: list[AgentStep] = []
        changed_files: set[str] = set()
        skill = self.skills.get(skill_name)

        if skill_name and skill is None:
            logger.warning(
                "agent_skill_not_found",
                run_id=run_id,
                skill=skill_name,
            )
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

        with request_context(agent_run_id=run_id):
            for index in range(self.max_steps):
                try:
                    decision = self.planner.decide(
                        task=task,
                        context=context,
                        history=history,
                        allowed_tools=allowed_tools,
                        skill=skill,
                    )
                except Exception as exc:
                    logger.error("agent_planner_failed", error=str(exc), step=index)
                    return AgentRunResult(
                        run_id=run_id,
                        task=task,
                        status="blocked",
                        selected_context=context,
                        steps=history,
                        summary=f"Planner failed: {exc}",
                        changed_files=sorted(changed_files),
                        skill=skill.name if skill else None,
                    )

                if decision.tool not in allowed_tools:
                    step = AgentStep(index=index, decision=decision)
                    history.append(step)
                    logger.warning(
                        "agent_tool_blocked",
                        step=index,
                        tool=decision.tool.value,
                        reason="tool_not_allowed_for_run",
                    )
                    return AgentRunResult(
                        run_id=run_id,
                        task=task,
                        status="blocked",
                        selected_context=context,
                        steps=history,
                        summary=f"Tool '{decision.tool.value}' is not allowed for this run.",
                        changed_files=sorted(changed_files),
                        skill=skill.name if skill else None,
                    )

                if decision.tool == AgentTool.FINISH:
                    history.append(AgentStep(index=index, decision=decision))
                    summary = str(
                        decision.arguments.get("summary")
                        or decision.reason
                        or "Agent run completed."
                    )
                    logger.info(
                        "agent_run_completed",
                        step=index,
                        changed_files=sorted(changed_files),
                    )
                    return AgentRunResult(
                        run_id=run_id,
                        task=task,
                        status="completed",
                        selected_context=context,
                        steps=history,
                        summary=summary,
                        changed_files=sorted(changed_files),
                        skill=skill.name if skill else None,
                    )

                result = self.registry.execute(decision.tool, decision.arguments)
                history.append(
                    AgentStep(
                        index=index,
                        decision=decision,
                        result=result,
                    )
                )
                logger.info(
                    "agent_tool_executed",
                    step=index,
                    tool=decision.tool.value,
                    success=result.success,
                    metadata=result.metadata,
                )

                if decision.tool == AgentTool.WRITE_FILE and result.success:
                    changed_path = result.metadata.get("path")
                    if isinstance(changed_path, str):
                        changed_files.add(changed_path)

                if result.metadata.get("blocked"):
                    return AgentRunResult(
                        run_id=run_id,
                        task=task,
                        status="blocked",
                        selected_context=context,
                        steps=history,
                        summary=result.output,
                        changed_files=sorted(changed_files),
                        skill=skill.name if skill else None,
                    )

        logger.warning("agent_run_max_steps", max_steps=self.max_steps)
        return AgentRunResult(
            run_id=run_id,
            task=task,
            status="max_steps",
            selected_context=context,
            steps=history,
            summary=f"Agent reached the maximum of {self.max_steps} steps.",
            changed_files=sorted(changed_files),
            skill=skill.name if skill else None,
        )
