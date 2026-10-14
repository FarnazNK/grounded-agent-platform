"""LangGraph execution workflow for the local developer-agent harness.

The graph orchestrates decisions and tool execution; policy remains enforced by
the existing tool registry and never delegated to the LLM.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_agent.agent.models import AgentRunResult, AgentStep, AgentTool

if TYPE_CHECKING:
    from rag_agent.agent.harness import DeveloperAgentHarness
    from rag_agent.agent.models import ContextFile
    from rag_agent.agent.skills import AgentSkill


class RunState(TypedDict):
    history: list[AgentStep]
    changed_files: set[str]
    status: str
    summary: str


def execute_graph(
    harness: DeveloperAgentHarness,
    *,
    run_id: str,
    task: str,
    context: list[ContextFile],
    skill: AgentSkill | None,
    allowed_tools: list[AgentTool],
) -> AgentRunResult:
    """Execute the actual planner -> tool -> planner state graph."""
    from rag_agent.observability import get_logger, request_context

    logger = get_logger(__name__)

    def plan(state: RunState) -> dict:
        index = len(state["history"])
        try:
            decision = harness.planner.decide(
                task=task,
                context=context,
                history=state["history"],
                allowed_tools=allowed_tools,
                skill=skill,
            )
        except Exception as exc:
            logger.error("agent_planner_failed", error=str(exc), step=index)
            return {"status": "blocked", "summary": f"Planner failed: {exc}"}

        step = AgentStep(index=index, decision=decision)
        history = [*state["history"], step]
        if decision.tool not in allowed_tools:
            logger.warning(
                "agent_tool_blocked",
                step=index,
                tool=decision.tool.value,
                reason="tool_not_allowed_for_run",
            )
            return {
                "history": history,
                "status": "blocked",
                "summary": f"Tool '{decision.tool.value}' is not allowed for this run.",
            }
        if decision.tool == AgentTool.FINISH:
            logger.info(
                "agent_run_completed",
                step=index,
                changed_files=sorted(state["changed_files"]),
            )
            return {
                "history": history,
                "status": "completed",
                "summary": str(
                    decision.arguments.get("summary") or decision.reason or "Agent run completed."
                ),
            }
        return {"history": history}

    def execute(state: RunState) -> dict:
        history = list(state["history"])
        step = history[-1]
        result = harness.registry.execute(step.decision.tool, step.decision.arguments)
        history[-1] = AgentStep(index=step.index, decision=step.decision, result=result)
        logger.info(
            "agent_tool_executed",
            step=step.index,
            tool=step.decision.tool.value,
            success=result.success,
            metadata=result.metadata,
        )
        changed_files = set(state["changed_files"])
        if step.decision.tool == AgentTool.WRITE_FILE and result.success:
            changed_path = result.metadata.get("path")
            if isinstance(changed_path, str):
                changed_files.add(changed_path)
        if result.metadata.get("blocked"):
            return {
                "history": history,
                "changed_files": changed_files,
                "status": "blocked",
                "summary": result.output,
            }
        return {"history": history, "changed_files": changed_files}

    def after_plan(state: RunState) -> str:
        return "end" if state["status"] != "running" else "execute"

    def after_execute(state: RunState) -> str:
        if state["status"] != "running":
            return "end"
        return "end" if len(state["history"]) >= harness.max_steps else "plan"

    graph = StateGraph(RunState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", after_plan, {"execute": "execute", "end": END})
    graph.add_conditional_edges("execute", after_execute, {"plan": "plan", "end": END})
    compiled = graph.compile()

    initial: RunState = {
        "history": [],
        "changed_files": set(),
        "status": "running",
        "summary": "",
    }
    if harness.max_steps < 1:
        final = initial
    else:
        with request_context(agent_run_id=run_id):
            final = compiled.invoke(
                initial,
                config={"recursion_limit": max(25, 2 * harness.max_steps + 5)},
            )

    status = final["status"]
    summary = final["summary"]
    if status == "running":
        status = "max_steps"
        summary = f"Agent reached the maximum of {harness.max_steps} steps."
        logger.warning("agent_run_max_steps", max_steps=harness.max_steps)
    return AgentRunResult(
        run_id=run_id,
        task=task,
        status=status,
        selected_context=context,
        steps=final["history"],
        summary=summary,
        changed_files=sorted(final["changed_files"]),
        skill=skill.name if skill else None,
    )
