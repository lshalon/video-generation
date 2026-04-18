"""Tests for the git-based current_code_version helper."""

import subprocess
from pathlib import Path

import pytest

from video_generation.store.code_version import UNKNOWN_SHA, current_code_version


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    _git(["init", "--initial-branch=main"], cwd=tmp_path)
    _git(["config", "user.email", "test@example.com"], cwd=tmp_path)
    _git(["config", "user.name", "Test"], cwd=tmp_path)
    (tmp_path / "README.md").write_text("hi")
    _git(["add", "README.md"], cwd=tmp_path)
    _git(["commit", "-m", "initial"], cwd=tmp_path)
    return tmp_path


class TestCurrentCodeVersion:
    def test_clean_repo(self, git_repo: Path) -> None:
        version = current_code_version(repo_root=git_repo)
        assert version.git_sha != UNKNOWN_SHA
        assert len(version.git_sha) == 40
        assert version.git_dirty is False

    def test_dirty_repo(self, git_repo: Path) -> None:
        (git_repo / "README.md").write_text("modified")
        version = current_code_version(repo_root=git_repo)
        assert version.git_dirty is True

    def test_outside_repo_returns_unknown(self, tmp_path: Path) -> None:
        outside = tmp_path / "not-a-repo"
        outside.mkdir()
        version = current_code_version(repo_root=outside)
        assert version.git_sha == UNKNOWN_SHA
        assert version.git_dirty is True
