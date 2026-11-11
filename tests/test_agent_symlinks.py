from __future__ import annotations

import os

import pytest

from rag_agent.agent.context import RepositoryContextBuilder
from rag_agent.agent.policy import PolicyViolation, ToolPolicy, read_regular_file
from rag_agent.agent.tools import RepositoryToolbox

SECRET = "TOP-SECRET-TOKEN"


@pytest.fixture
def repo(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(SECRET, encoding="utf-8")

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (root / ".env").write_text(f"API_KEY={SECRET}\n", encoding="utf-8")
    # Harmless-looking names pointing outside the repo and at an excluded file.
    (root / "src" / "notes.md").symlink_to(outside / "secret.txt")
    (root / "src" / "config.py").symlink_to(root / ".env")
    (root / "linked_dir").symlink_to(outside, target_is_directory=True)
    return root


def toolbox(root, *, allow_writes=False):
    policy = ToolPolicy(allow_writes=allow_writes)
    return RepositoryToolbox(root, policy=policy, context_builder=RepositoryContextBuilder(root))


def test_context_builder_skips_symlinks(repo):
    builder = RepositoryContextBuilder(repo)

    paths = {p.relative_to(builder.repo_root).as_posix() for p in builder.iter_candidate_paths()}

    assert paths == {"src/app.py"}
    assert all(SECRET not in item.content for item in builder.build("notes config secret"))


def test_search_never_returns_symlinked_content(repo):
    result = toolbox(repo).search_code({"query": "TOP-SECRET"})

    assert result.success
    assert SECRET not in result.output


@pytest.mark.parametrize("path", ["src/notes.md", "src/config.py", "linked_dir/secret.txt"])
def test_read_file_rejects_symlinks(repo, path):
    result = toolbox(repo).read_file({"path": path})

    assert result.success is False
    assert result.metadata["blocked"] is True
    assert SECRET not in result.output


def test_write_file_cannot_write_through_a_symlink(repo, tmp_path):
    result = toolbox(repo, allow_writes=True).write_file(
        {"path": "linked_dir/pwned.txt", "content": "x"}
    )

    assert result.success is False
    assert not (tmp_path / "outside" / "pwned.txt").exists()


def test_read_refuses_a_file_swapped_for_a_symlink_after_validation(repo):
    target = repo / "src" / "app.py"
    validated = ToolPolicy().validate_path(repo, "src/app.py")
    target.unlink()
    target.symlink_to(repo.parent / "outside" / "secret.txt")

    with pytest.raises(OSError):
        read_regular_file(validated)


def test_validate_path_rejects_symlinks_even_inside_the_repo(repo):
    (repo / "alias").symlink_to(repo / "src", target_is_directory=True)

    with pytest.raises(PolicyViolation, match="Symlinks"):
        ToolPolicy().validate_path(repo, "alias/app.py")
    with pytest.raises(PolicyViolation, match="escapes"):
        ToolPolicy().validate_path(repo, "linked_dir/secret.txt")


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX only")
def test_regular_files_still_readable(repo):
    assert read_regular_file(repo / "src" / "app.py") == "print('hello')\n"
