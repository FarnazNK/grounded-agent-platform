from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentTool(StrEnum):
    LIST_FILES = "list_files"
    READ_FILE = "read_file"
    SEARCH_CODE = "search_code"
    WRITE_FILE = "write_file"
    RUN_CHECK = "run_check"
    FINISH = "finish"


class ContextFile(BaseModel):
    path: str
    score: float
    content: str
    truncated: bool = False


class AgentDecision(BaseModel):
    tool: AgentTool
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ToolResult(BaseModel):
    tool: AgentTool
    success: bool
    output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    index: int
    decision: AgentDecision
    result: ToolResult | None = None


class AgentRunResult(BaseModel):
    run_id: str
    task: str
    status: Literal["completed", "blocked", "max_steps"]
    selected_context: list[ContextFile] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    skill: str | None = None


class TeamRole(StrEnum):
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"


class RoleRun(BaseModel):
    role: TeamRole
    round: int
    result: AgentRunResult


class TeamRunResult(BaseModel):
    run_id: str
    task: str
    status: Literal["completed", "blocked", "max_steps", "changes_requested"]
    plan: str = ""
    review_verdict: Literal["approve", "request_changes"] | None = None
    review_rounds: int = 0
    total_steps: int = 0
    runs: list[RoleRun] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    summary: str = ""
