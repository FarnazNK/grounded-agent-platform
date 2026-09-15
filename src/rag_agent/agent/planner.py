from __future__ import annotations

import json
from typing import Any, Protocol

from rag_agent.agent.models import AgentDecision, AgentStep, AgentTool, ContextFile
from rag_agent.agent.skills import AgentSkill
from rag_agent.config import Settings


class AgentPlanner(Protocol):
    def decide(
        self,
        *,
        task: str,
        context: list[ContextFile],
        history: list[AgentStep],
        allowed_tools: list[AgentTool],
        skill: AgentSkill | None,
    ) -> AgentDecision: ...


class DeterministicPlanner:
    """Offline planner used for CI and context-only dry runs."""

    def decide(
        self,
        *,
        task: str,
        context: list[ContextFile],
        history: list[AgentStep],
        allowed_tools: list[AgentTool],
        skill: AgentSkill | None,
    ) -> AgentDecision:
        files = ", ".join(item.path for item in context[:5]) or "no matching files"
        return AgentDecision(
            tool=AgentTool.FINISH,
            arguments={
                "summary": (
                    "Deterministic dry run completed. "
                    f"Selected repository context for '{task}': {files}."
                )
            },
            reason="Remote model execution is disabled in deterministic mode.",
        )


class LLMToolPlanner:
    """JSON tool planner backed by the configured Anthropic or OpenAI chat model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            client_cls: Any = ChatAnthropic
            self._client = client_cls(
                model=settings.llm_model,
                temperature=0,
                timeout=settings.llm_request_timeout_seconds,
                max_tokens=settings.llm_max_output_tokens,
            )
        elif settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            client_cls = ChatOpenAI
            self._client = client_cls(
                model=settings.llm_model,
                temperature=0,
                timeout=settings.llm_request_timeout_seconds,
                max_tokens=settings.llm_max_output_tokens,
            )
        else:
            raise ValueError("LLMToolPlanner requires anthropic or openai provider.")

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _strip_fence(value: str) -> str:
        text = value.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def decide(
        self,
        *,
        task: str,
        context: list[ContextFile],
        history: list[AgentStep],
        allowed_tools: list[AgentTool],
        skill: AgentSkill | None,
    ) -> AgentDecision:
        context_payload = [
            {
                "path": item.path,
                "score": item.score,
                "content": item.content[:6000],
            }
            for item in context[:8]
        ]
        history_payload = [
            {
                "tool": step.decision.tool.value,
                "arguments": step.decision.arguments,
                "reason": step.decision.reason,
                "success": step.result.success if step.result else None,
                "output": step.result.output[-4000:] if step.result else None,
            }
            for step in history[-8:]
        ]

        system = (
            "You are a repository-aware software engineering agent. "
            "Choose exactly one next action. Work incrementally: inspect relevant code, "
            "make the smallest justified change, then verify it. "
            "Never request tools that are not in allowed_tools. "
            "Return only a JSON object with keys: tool, arguments, reason. "
            "For finish, set arguments.summary to a concise description of the result. "
            "For write_file, provide the complete replacement file content. "
            "For run_check, use only normal test/lint/typecheck/build commands; "
            "the runtime will independently enforce an allowlist."
        )
        if skill is not None:
            system += f"\nActive skill: {skill.name}\n{skill.instructions}"

        user = json.dumps(
            {
                "task": task,
                "allowed_tools": [tool.value for tool in allowed_tools],
                "repository_context": context_payload,
                "history": history_payload,
            },
            ensure_ascii=False,
        )
        response = self._client.invoke(
            [
                ("system", system),
                ("human", user),
            ]
        )
        payload = json.loads(self._strip_fence(self._response_text(response)))
        return AgentDecision.model_validate(payload)


def build_planner(settings: Settings) -> AgentPlanner:
    if settings.llm_provider == "deterministic":
        return DeterministicPlanner()
    return LLMToolPlanner(settings)
