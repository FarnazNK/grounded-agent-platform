from __future__ import annotations

from collections import deque

from rag_agent.agent import team as team_module
from rag_agent.agent.models import AgentDecision, AgentTool, TeamRole
from rag_agent.agent.planner import DeterministicPlanner
from rag_agent.agent.policy import ToolPolicy
from rag_agent.agent.team import DeveloperAgentTeam
from rag_agent.tracing import Tracer


class RolePlanner:
    """Scripted planner that answers differently per team role (selected by skill name)."""

    def __init__(self, scripts: dict[str, list[AgentDecision]]) -> None:
        self.scripts = {name: deque(items) for name, items in scripts.items()}
        self.tasks: dict[str, list[str]] = {name: [] for name in scripts}

    def decide(self, *, task, skill, history, **_kwargs):
        name = skill.name
        if not history:
            self.tasks[name].append(task)
        return self.scripts[name].popleft()


def finish(summary: str, **extra) -> AgentDecision:
    return AgentDecision(tool=AgentTool.FINISH, arguments={"summary": summary, **extra})


def write(path: str, content: str) -> AgentDecision:
    return AgentDecision(tool=AgentTool.WRITE_FILE, arguments={"path": path, "content": content})


def make_repo(tmp_path):
    (tmp_path / "service.py").write_text("value = 1\n", encoding="utf-8")
    return tmp_path


def test_team_plans_codes_and_gets_approval(tmp_path):
    repo = make_repo(tmp_path)
    planner = RolePlanner(
        {
            "team-planner": [finish("1. set value to 2 in service.py")],
            "team-coder": [write("service.py", "value = 2\n"), finish("Set value to 2.")],
            "team-reviewer": [finish("Looks correct.", verdict="approve")],
        }
    )

    result = DeveloperAgentTeam(repo, planner=planner, policy=ToolPolicy(allow_writes=True)).run(
        "change value to 2"
    )

    assert result.status == "completed"
    assert result.plan == "1. set value to 2 in service.py"
    assert result.review_verdict == "approve"
    assert result.review_rounds == 1
    assert result.changed_files == ["service.py"]
    assert [run.role for run in result.runs] == [
        TeamRole.PLANNER,
        TeamRole.CODER,
        TeamRole.REVIEWER,
    ]
    assert result.total_steps == 4
    assert (repo / "service.py").read_text(encoding="utf-8") == "value = 2\n"
    assert "1. set value to 2" in planner.tasks["team-coder"][0]


def test_reviewer_feedback_is_sent_back_to_the_coder(tmp_path):
    repo = make_repo(tmp_path)
    planner = RolePlanner(
        {
            "team-planner": [finish("plan")],
            "team-coder": [
                write("service.py", "value = 3\n"),
                finish("first attempt"),
                write("service.py", "value = 2\n"),
                finish("second attempt"),
            ],
            "team-reviewer": [
                finish("Value must be 2, not 3.", verdict="request_changes"),
                finish("Fixed.", verdict="approve"),
            ],
        }
    )

    result = DeveloperAgentTeam(repo, planner=planner, policy=ToolPolicy(allow_writes=True)).run(
        "change value to 2"
    )

    assert result.status == "completed"
    assert result.review_rounds == 2
    assert "Value must be 2, not 3." in planner.tasks["team-coder"][1]
    assert (repo / "service.py").read_text(encoding="utf-8") == "value = 2\n"


def test_team_stops_when_reviewer_never_approves(tmp_path):
    repo = make_repo(tmp_path)
    planner = RolePlanner(
        {
            "team-planner": [finish("plan")],
            "team-coder": [finish("attempt 1"), finish("attempt 2")],
            "team-reviewer": [
                finish("still wrong", verdict="request_changes"),
                finish("no verdict given"),
            ],
        }
    )

    result = DeveloperAgentTeam(repo, planner=planner, max_review_rounds=2).run("task")

    assert result.status == "changes_requested"
    assert result.review_verdict == "request_changes"
    assert result.review_rounds == 2


def test_reviewer_cannot_write_even_when_writes_are_enabled(tmp_path):
    repo = make_repo(tmp_path)
    planner = RolePlanner(
        {
            "team-planner": [finish("plan")],
            "team-coder": [finish("no change needed")],
            "team-reviewer": [write("service.py", "value = 99\n")],
        }
    )

    result = DeveloperAgentTeam(repo, planner=planner, policy=ToolPolicy(allow_writes=True)).run(
        "task"
    )

    assert result.status == "blocked"
    assert (repo / "service.py").read_text(encoding="utf-8") == "value = 1\n"


def test_planner_is_read_only(tmp_path):
    repo = make_repo(tmp_path)
    planner = RolePlanner({"team-planner": [write("service.py", "value = 5\n")]})

    result = DeveloperAgentTeam(repo, planner=planner, policy=ToolPolicy(allow_writes=True)).run(
        "task"
    )

    assert result.status == "blocked"
    assert [run.role for run in result.runs] == [TeamRole.PLANNER]
    assert (repo / "service.py").read_text(encoding="utf-8") == "value = 1\n"


def test_team_step_budget_is_enforced(tmp_path):
    repo = make_repo(tmp_path)
    read = AgentDecision(tool=AgentTool.READ_FILE, arguments={"path": "service.py"})
    planner = RolePlanner(
        {
            "team-planner": [read, read, finish("plan")],
            "team-coder": [read, read, read],
        }
    )

    result = DeveloperAgentTeam(repo, planner=planner, max_total_steps=5).run("task")

    assert result.status == "max_steps"
    assert result.total_steps == 5


def test_deterministic_dry_run_completes(tmp_path):
    repo = make_repo(tmp_path)

    result = DeveloperAgentTeam(repo, planner=DeterministicPlanner()).run("review service")

    assert result.status == "completed"
    assert result.review_verdict == "approve"
    assert result.changed_files == []


class RecordingTracer(Tracer):
    def __init__(self) -> None:
        self.names: list[str] = []
        self.scores: list[tuple[str, float]] = []

    def observe(self, name, **_kwargs):
        self.names.append(name)
        return super().observe(name)

    def score(self, name, value, *, comment=None):
        self.scores.append((name, value))


def test_team_run_emits_nested_agent_and_tool_spans(tmp_path, monkeypatch):
    from rag_agent.agent import harness as harness_module

    tracer = RecordingTracer()
    monkeypatch.setattr(team_module, "get_tracer", lambda: tracer)
    monkeypatch.setattr(harness_module, "get_tracer", lambda: tracer)
    repo = make_repo(tmp_path)
    planner = RolePlanner(
        {
            "team-planner": [finish("plan")],
            "team-coder": [write("service.py", "value = 2\n"), finish("done")],
            "team-reviewer": [finish("ok", verdict="approve")],
        }
    )

    DeveloperAgentTeam(repo, planner=planner, policy=ToolPolicy(allow_writes=True)).run("t")

    assert tracer.names == [
        "dev-team",
        "agent.planner",
        "agent.coder",
        "tool.write_file",
        "agent.reviewer",
    ]
    assert tracer.scores == [("review_approved", 1.0)]
