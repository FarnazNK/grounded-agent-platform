from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from rag_agent.agent.models import AgentTool


class AgentSkill(BaseModel):
    name: str
    description: str = ""
    instructions: str
    allowed_tools: list[AgentTool] = Field(default_factory=list)


class SkillRegistry:
    """Load reusable agent behavior from declarative YAML skill files."""

    def __init__(self, skills: list[AgentSkill] | None = None) -> None:
        self._skills = {skill.name: skill for skill in skills or []}

    @classmethod
    def from_directory(cls, directory: str | Path) -> SkillRegistry:
        root = Path(directory)
        if not root.exists():
            return cls()

        skills: list[AgentSkill] = []
        for path in sorted(root.glob("*.yaml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            skills.append(AgentSkill.model_validate(payload))
        return cls(skills)

    def get(self, name: str | None) -> AgentSkill | None:
        if not name:
            return None
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills)
