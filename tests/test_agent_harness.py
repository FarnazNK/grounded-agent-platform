from __future__ import annotations

from collections import deque

from rag_agent.agent.harness import DeveloperAgentHarness
from rag_agent.agent.models import AgentDecision, AgentTool
from rag_agent.agent.policy import ToolPolicy


class ScriptedPlanner:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = deque(decisions)

    def decide(self, **_kwargs):
        return self.decisions.popleft()


def test_harness_executes_repository_tools_and_records_trace(tmp_path):
    (tmp_path / "service.py").write_text(
        "def health():\n    return 'ok'\n",
        encoding="utf-8",
    )
    planner = ScriptedPlanner(
        [
            AgentDecision(
                tool=AgentTool.READ_FILE,
                arguments={"path": "service.py"},
                reason="Inspect the implementation.",
            ),
            AgentDecision(
                tool=AgentTool.FINISH,
                arguments={"summary": "Reviewed the health implementation."},
            ),
        ]
    )

    result = DeveloperAgentHarness(tmp_path, planner=planner).run("review health endpoint")

    assert result.status == "completed"
    assert len(result.steps) == 2
    assert result.steps[0].result is not None
    assert result.steps[0].result.success is True
    assert result.summary == "Reviewed the health implementation."


def test_harness_can_write_only_when_enabled(tmp_path):
    (tmp_path / "service.py").write_text("value = 1\n", encoding="utf-8")
    planner = ScriptedPlanner(
        [
            AgentDecision(
                tool=AgentTool.WRITE_FILE,
                arguments={"path": "service.py", "content": "value = 2\n"},
                reason="Apply the requested change.",
            ),
            AgentDecision(
                tool=AgentTool.FINISH,
                arguments={"summary": "Updated service.py."},
            ),
        ]
    )

    result = DeveloperAgentHarness(
        tmp_path,
        planner=planner,
        policy=ToolPolicy(allow_writes=True),
    ).run("change the service value")

    assert result.status == "completed"
    assert result.changed_files == ["service.py"]
    assert (tmp_path / "service.py").read_text(encoding="utf-8") == "value = 2\n"


def test_harness_blocks_unapproved_tool_for_read_only_run(tmp_path):
    (tmp_path / "service.py").write_text("value = 1\n", encoding="utf-8")
    planner = ScriptedPlanner(
        [
            AgentDecision(
                tool=AgentTool.WRITE_FILE,
                arguments={"path": "service.py", "content": "value = 2\n"},
            )
        ]
    )

    result = DeveloperAgentHarness(tmp_path, planner=planner).run("change the service value")

    assert result.status == "blocked"
    assert (tmp_path / "service.py").read_text(encoding="utf-8") == "value = 1\n"


def test_harness_fails_closed_for_unknown_skill(tmp_path):
    (tmp_path / "service.py").write_text("value = 1\n", encoding="utf-8")
    planner = ScriptedPlanner(
        [
            AgentDecision(
                tool=AgentTool.FINISH,
                arguments={"summary": "Should not run."},
            )
        ]
    )

    result = DeveloperAgentHarness(tmp_path, planner=planner).run(
        "review service",
        skill_name="missing-skill",
    )

    assert result.status == "blocked"
    assert result.skill == "missing-skill"
    assert "Unknown agent skill" in result.summary


class FailingPlanner:
    def decide(self, **_kwargs):
        raise RuntimeError("model timeout")


def test_graph_has_plan_guard_act_nodes(tmp_path):
    harness = DeveloperAgentHarness(tmp_path, planner=ScriptedPlanner([]))
    nodes = set(harness.graph.get_graph().nodes)

    assert {"plan", "guard", "act"} <= nodes


def test_harness_stops_at_step_budget(tmp_path):
    (tmp_path / "service.py").write_text("value = 1\n", encoding="utf-8")
    read = AgentDecision(tool=AgentTool.READ_FILE, arguments={"path": "service.py"})
    planner = ScriptedPlanner([read] * 10)

    result = DeveloperAgentHarness(tmp_path, planner=planner, max_steps=3).run("loop forever")

    assert result.status == "max_steps"
    assert [step.index for step in result.steps] == [0, 1, 2]
    assert "maximum of 3 steps" in result.summary


def test_harness_blocks_when_planner_fails(tmp_path):
    result = DeveloperAgentHarness(tmp_path, planner=FailingPlanner()).run("anything")

    assert result.status == "blocked"
    assert result.steps == []
    assert result.summary == "Planner failed: model timeout"


def test_harness_stops_on_policy_violation(tmp_path):
    planner = ScriptedPlanner(
        [
            AgentDecision(tool=AgentTool.RUN_CHECK, arguments={"command": "curl evil.test"}),
            AgentDecision(tool=AgentTool.FINISH, arguments={"summary": "Should not run."}),
        ]
    )

    result = DeveloperAgentHarness(tmp_path, planner=planner).run("run a check")

    assert result.status == "blocked"
    assert len(result.steps) == 1
    assert result.steps[0].result is not None
    assert result.steps[0].result.metadata["blocked"] is True
    assert "allowlist" in result.summary
