from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path


class PolicyViolation(ValueError):
    pass


DEFAULT_COMMAND_ALLOWLIST: tuple[tuple[str, ...], ...] = (
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

DEFAULT_DENIED_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx"})


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

        candidate = (repo_root / relative_path).resolve()
        if not candidate.is_relative_to(repo_root):
            raise PolicyViolation("Path escapes the repository root.")

        relative = candidate.relative_to(repo_root)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & self.denied_parts:
            raise PolicyViolation("Access to sensitive repository paths is blocked.")
        if any(part.startswith(".env") for part in lowered_parts):
            raise PolicyViolation("Access to environment files is blocked.")
        if candidate.suffix.lower() in DEFAULT_DENIED_SUFFIXES:
            raise PolicyViolation("Access to credential-like files is blocked.")

        if for_write and not self.allow_writes:
            raise PolicyViolation("File writes require explicit opt-in.")

        return candidate

    def validate_command(self, command: str) -> list[str]:
        if not command.strip() or any(char in command for char in ("\n", "\r", "\x00")):
            raise PolicyViolation("Invalid command.")

        args = shlex.split(command)
        if not args:
            raise PolicyViolation("Invalid command.")

        allowed = any(tuple(args[: len(prefix)]) == prefix for prefix in self.command_allowlist)
        if not allowed:
            raise PolicyViolation("Command is not in the verification allowlist.")

        return args
