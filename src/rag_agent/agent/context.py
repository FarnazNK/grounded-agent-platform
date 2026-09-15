from __future__ import annotations

import re
from pathlib import Path

from rag_agent.agent.models import ContextFile

_RAW_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

DEFAULT_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "vendor",
    }
)

DEFAULT_IGNORED_PATH_PREFIXES = (
    "benchmarks/results/",
    "data/agent_evals/",
)

DEFAULT_SENSITIVE_PARTS = frozenset(
    {
        ".aws",
        ".gnupg",
        ".ssh",
        "credentials",
        "secrets",
    }
)

DEFAULT_TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".css",
        ".dart",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)

_ALWAYS_TEXT_FILENAMES = frozenset(
    {
        "Dockerfile",
        "Gemfile",
        "Makefile",
        "Procfile",
        "Rakefile",
    }
)

_SOURCE_PREFIXES = (
    "app/",
    "lib/",
    "src/",
)


def _normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    return token


def _tokens(value: str) -> set[str]:
    expanded = _CAMEL_BOUNDARY_RE.sub(" ", value.replace("_", " "))
    return {
        normalized
        for raw in _RAW_TOKEN_RE.findall(expanded)
        if len(normalized := _normalize_token(raw)) > 1
    }


def _overlap_count(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    score = 0.0
    for query_token in query_tokens:
        if query_token in candidate_tokens:
            score += 1.0
            continue
        if len(query_token) < 4:
            continue
        if any(
            len(candidate) >= 4
            and (query_token.startswith(candidate) or candidate.startswith(query_token))
            for candidate in candidate_tokens
        ):
            score += 0.5
    return score


class RepositoryContextBuilder:
    """Select a compact set of repository files relevant to a developer task."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        max_file_bytes: int = 200_000,
        max_context_chars: int = 40_000,
        ignored_dirs: frozenset[str] = DEFAULT_IGNORED_DIRS,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        if not self.repo_root.is_dir():
            raise ValueError(f"Repository root does not exist: {self.repo_root}")
        self.max_file_bytes = max_file_bytes
        self.max_context_chars = max_context_chars
        self.ignored_dirs = ignored_dirs

    def resolve_path(self, relative_path: str | Path) -> Path:
        candidate = (self.repo_root / relative_path).resolve()
        if not candidate.is_relative_to(self.repo_root):
            raise ValueError("Path escapes the repository root.")
        return candidate

    def iter_candidate_paths(self) -> list[Path]:
        candidates: list[Path] = []
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.repo_root)
            relative_posix = relative.as_posix()
            lowered_parts = {part.lower() for part in relative.parts}
            if any(part in self.ignored_dirs for part in relative.parts):
                continue
            if any(relative_posix.startswith(prefix) for prefix in DEFAULT_IGNORED_PATH_PREFIXES):
                continue
            if lowered_parts & DEFAULT_SENSITIVE_PARTS:
                continue
            if path.name.lower().startswith((".env", "secret")):
                continue
            is_supported = (
                path.name in _ALWAYS_TEXT_FILENAMES or path.suffix.lower() in DEFAULT_TEXT_SUFFIXES
            )
            if not is_supported:
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            candidates.append(path)
        return sorted(candidates)

    def _read(self, path: Path) -> tuple[str, bool]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        max_chars = min(self.max_file_bytes, self.max_context_chars)
        if len(raw) <= max_chars:
            return raw, False
        return raw[:max_chars], True

    def build(self, task: str, *, limit: int = 12) -> list[ContextFile]:
        query_tokens = _tokens(task)
        ranked: list[ContextFile] = []

        for path in self.iter_candidate_paths():
            relative = path.relative_to(self.repo_root).as_posix()
            try:
                content, truncated = self._read(path)
            except OSError:
                continue

            path_tokens = _tokens(relative.replace("/", " "))
            content_tokens = _tokens(content[:20_000])
            path_overlap = _overlap_count(query_tokens, path_tokens)
            content_overlap = _overlap_count(query_tokens, content_tokens)

            score = path_overlap * 4 + content_overlap
            if relative.startswith(_SOURCE_PREFIXES):
                score += 2.0
            elif relative.startswith(("tests/", "test/", "spec/")):
                score += 0.5
            elif relative.startswith(("docs/", "data/", "benchmarks/")):
                score -= 1.0

            if path.name.lower() in {"readme.md", "pyproject.toml", "package.json", "gemfile"}:
                score += 0.25
            if score <= 0:
                continue

            ranked.append(
                ContextFile(
                    path=relative,
                    score=score,
                    content=content,
                    truncated=truncated,
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.path))

        selected: list[ContextFile] = []
        used_chars = 0
        for item in ranked:
            if len(selected) >= limit:
                break
            remaining = self.max_context_chars - used_chars
            if remaining <= 0:
                break
            if len(item.content) > remaining:
                item = item.model_copy(
                    update={
                        "content": item.content[:remaining],
                        "truncated": True,
                    }
                )
            selected.append(item)
            used_chars += len(item.content)

        return selected
