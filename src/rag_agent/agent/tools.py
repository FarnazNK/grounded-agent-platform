from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_agent.agent.context import RepositoryContextBuilder
from rag_agent.agent.models import AgentTool, ToolResult
from rag_agent.agent.policy import PolicyViolation, ToolPolicy

ToolHandler = Callable[[dict[str, Any]], ToolResult]


class ToolRegistry:
    """Small registry so repository tools can be extended without changing the harness."""

    def __init__(self) -> None:
        self._handlers: dict[AgentTool, ToolHandler] = {}

    def register(self, tool: AgentTool, handler: ToolHandler) -> None:
        self._handlers[tool] = handler

    @property
    def names(self) -> list[AgentTool]:
        return sorted(self._handlers, key=str)

    def execute(self, tool: AgentTool, arguments: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(tool)
        if handler is None:
            return ToolResult(tool=tool, success=False, output="Tool is not registered.")
        return handler(arguments)


class RepositoryToolbox:
    def __init__(
        self,
        repo_root: str | Path,
        *,
        policy: ToolPolicy | None = None,
        context_builder: RepositoryContextBuilder | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.policy = policy or ToolPolicy()
        self.context_builder = context_builder or RepositoryContextBuilder(self.repo_root)

    def registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(AgentTool.LIST_FILES, self.list_files)
        registry.register(AgentTool.READ_FILE, self.read_file)
        registry.register(AgentTool.SEARCH_CODE, self.search_code)
        registry.register(AgentTool.WRITE_FILE, self.write_file)
        registry.register(AgentTool.RUN_CHECK, self.run_check)
        return registry

    def list_files(self, arguments: dict[str, Any]) -> ToolResult:
        limit = max(1, min(int(arguments.get("limit", 200)), 1000))
        files = [
            path.relative_to(self.repo_root).as_posix()
            for path in self.context_builder.iter_candidate_paths()[:limit]
        ]
        return ToolResult(
            tool=AgentTool.LIST_FILES,
            success=True,
            output="\n".join(files),
            metadata={"count": len(files)},
        )

    def read_file(self, arguments: dict[str, Any]) -> ToolResult:
        relative_path = str(arguments.get("path", ""))
        try:
            path = self.policy.validate_path(self.repo_root, relative_path)
            if not path.is_file():
                return ToolResult(
                    tool=AgentTool.READ_FILE,
                    success=False,
                    output="File does not exist.",
                )
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > self.context_builder.max_file_bytes:
                content = content[: self.context_builder.max_file_bytes]
                truncated = True
            else:
                truncated = False
            return ToolResult(
                tool=AgentTool.READ_FILE,
                success=True,
                output=content,
                metadata={"path": relative_path, "truncated": truncated},
            )
        except (OSError, PolicyViolation) as exc:
            return ToolResult(
                tool=AgentTool.READ_FILE,
                success=False,
                output=str(exc),
                metadata={"blocked": isinstance(exc, PolicyViolation)},
            )

    def search_code(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        limit = max(1, min(int(arguments.get("limit", 30)), 100))
        if not query:
            return ToolResult(
                tool=AgentTool.SEARCH_CODE,
                success=False,
                output="Search query is required.",
            )

        query_lower = query.lower()
        matches: list[str] = []
        for path in self.context_builder.iter_candidate_paths():
            relative = path.relative_to(self.repo_root).as_posix()
            try:
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                ):
                    if query_lower in line.lower():
                        matches.append(f"{relative}:{line_number}: {line[:240]}")
                        if len(matches) >= limit:
                            break
            except OSError:
                continue
            if len(matches) >= limit:
                break

        return ToolResult(
            tool=AgentTool.SEARCH_CODE,
            success=True,
            output="\n".join(matches),
            metadata={"count": len(matches), "query": query},
        )

    def write_file(self, arguments: dict[str, Any]) -> ToolResult:
        relative_path = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))
        try:
            path = self.policy.validate_path(self.repo_root, relative_path, for_write=True)
            encoded = content.encode("utf-8")
            if len(encoded) > self.policy.max_write_bytes:
                raise PolicyViolation("Write exceeds the configured size limit.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(
                tool=AgentTool.WRITE_FILE,
                success=True,
                output=f"Wrote {len(encoded)} bytes to {relative_path}.",
                metadata={"path": relative_path, "bytes": len(encoded)},
            )
        except (OSError, PolicyViolation) as exc:
            return ToolResult(
                tool=AgentTool.WRITE_FILE,
                success=False,
                output=str(exc),
                metadata={"blocked": isinstance(exc, PolicyViolation)},
            )

    def run_check(self, arguments: dict[str, Any]) -> ToolResult:
        command = str(arguments.get("command", ""))
        try:
            args = self.policy.validate_command(command, repo_root=self.repo_root)
        except PolicyViolation as exc:
            return ToolResult(
                tool=AgentTool.RUN_CHECK,
                success=False,
                output=str(exc),
                metadata={"blocked": True},
            )

        safe_env = {
            key: value
            for key, value in os.environ.items()
            if key in {"CI", "HOME", "LANG", "LC_ALL", "PATH", "PYTHONPATH", "TMPDIR"}
        }
        try:
            completed = subprocess.run(
                args,
                cwd=self.repo_root,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=self.policy.command_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(
                tool=AgentTool.RUN_CHECK,
                success=False,
                output=str(exc),
                metadata={"command": command},
            )

        output = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        return ToolResult(
            tool=AgentTool.RUN_CHECK,
            success=completed.returncode == 0,
            output=output[-20_000:],
            metadata={"command": command, "returncode": completed.returncode},
        )
