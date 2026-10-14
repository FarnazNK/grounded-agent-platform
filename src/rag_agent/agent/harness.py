from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rag_agent.agent.context import RepositoryContextBuilder
from rag_agent.agent.models import AgentRunResult, AgentTool
from rag_agent.agent.planner import AgentPlanner
from rag_agent.agent.policy import ToolPolicy
from rag_agent.agent.skills import SkillRegistry
from rag_agent.agent.tools import RepositoryToolbox
from rag_agent.observability import get_logger

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

        from rag_agent.agent.graph_runtime import execute_graph

        return execute_graph(
            self,
            run_id=run_id,
            task=task,
            context=context,
            skill=skill,
            allowed_tools=allowed_tools,
        )
