from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path


class PolicyViolation(ValueError):
    pass


DEFAULT_COMMAND_ALLOWLIST: tuple[tuple[str, ...], ...] = (
    ("pytest", "-q"),
    ("python", "-m", "pytest", "-q"),
    ("pytest",),
    ("python", "-m", "pytest"),
    ("ruff", "check"),
    ("ruff", "format", "--check"),
    ("python", "-m", "ruff", "check"),
    ("mypy",),
    ("npm", "test"),
    ("npm", "run", "test"),
    ("npm", "run", "lint"),
    ("npm", "run", "typecheck"),
    ("npm", "run", "build"),
    ("pnpm", "test"),
    ("pnpm", "lint"),
    ("yarn", "test"),
    ("bundle", "exec", "rspec"),
    ("bin/rails", "test"),
)

DEFAULT_DENIED_PARTS = frozenset(
    {
        ".aws",
        ".git",
        ".gnupg",
        ".ssh",
        "credentials",
        "id_ed25519",
        "id_rsa",
        "secrets",
    }
)

DEFAULT_DENIED_FILENAMES = frozenset(
    {
        "credentials.json",
        "credentials.yaml",
        "credentials.yml",
        "master.key",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    }
)

DEFAULT_DENIED_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx"})
DEFAULT_DENIED_ENDINGS = (".tfvars", ".tfvars.json")


def is_sensitive_path(
    relative_path: str | Path,
    *,
    denied_parts: frozenset[str] = DEFAULT_DENIED_PARTS,
) -> bool:
    relative = Path(relative_path)
    lowered_parts = {part.lower() for part in relative.parts}

    if lowered_parts & denied_parts:
        return True

    name = relative.name.lower()
    if name.startswith(".env"):
        return True
    if name in DEFAULT_DENIED_FILENAMES:
        return True
    if name.endswith(DEFAULT_DENIED_ENDINGS):
        return True

    return relative.suffix.lower() in DEFAULT_DENIED_SUFFIXES


@dataclass(frozen=True)
class ToolPolicy:
    allow_writes: bool = False
    command_timeout_seconds: float = 60.0
    max_write_bytes: int = 500_000
    command_allowlist: tuple[tuple[str, ...], ...] = DEFAULT_COMMAND_ALLOWLIST
    denied_parts: frozenset[str] = field(default_factory=lambda: DEFAULT_DENIED_PARTS)

    def validate_path(
        self,
        repo_root: Path,
        relative_path: str,
        *,
        for_write: bool = False,
    ) -> Path:
        if not relative_path or "\x00" in relative_path:
            raise PolicyViolation("Invalid repository path.")

        repo_root = repo_root.resolve()
        candidate = (repo_root / relative_path).resolve()
        if not candidate.is_relative_to(repo_root):
            raise PolicyViolation("Path escapes the repository root.")

        relative = candidate.relative_to(repo_root)
        if is_sensitive_path(relative, denied_parts=self.denied_parts):
            raise PolicyViolation("Access to sensitive repository paths is blocked.")

        if for_write and not self.allow_writes:
            raise PolicyViolation("File writes require explicit opt-in.")

        return candidate

    def _validate_command_path(self, token: str, repo_root: Path | None) -> None:
        path_token = token.split("::", 1)[0]
        if not path_token:
            raise PolicyViolation("Verification command contains an invalid path.")

        relative = Path(path_token)
        if relative.is_absolute() or ".." in relative.parts:
            raise PolicyViolation("Verification command path escapes the repository root.")
        if is_sensitive_path(relative, denied_parts=self.denied_parts):
            raise PolicyViolation("Verification command cannot access sensitive paths.")

        if repo_root is None:
            return

        repo_root = repo_root.resolve()
        candidate = (repo_root / relative).resolve()
        if not candidate.is_relative_to(repo_root):
            raise PolicyViolation("Verification command path escapes the repository root.")
        if not candidate.exists():
            raise PolicyViolation("Verification command paths must exist in the repository.")

    def validate_command(
        self,
        command: str,
        *,
        repo_root: Path | None = None,
    ) -> list[str]:
        if not command.strip() or any(char in command for char in ("\n", "\r", "\x00")):
            raise PolicyViolation("Invalid command.")

        args = shlex.split(command)
        if not args:
            raise PolicyViolation("Invalid command.")

        matching_prefixes = [
            prefix for prefix in self.command_allowlist if tuple(args[: len(prefix)]) == prefix
        ]
        if not matching_prefixes:
            raise PolicyViolation("Command is not in the verification allowlist.")

        prefix = max(matching_prefixes, key=len)
        for extra in args[len(prefix) :]:
            if extra.startswith("-"):
                raise PolicyViolation("Only repository paths may follow a verification command.")
            self._validate_command_path(extra, repo_root)

        return args
