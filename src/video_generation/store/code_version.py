"""Capture the current code version (git SHA + dirty flag).

This snapshot is recorded inside every run manifest so that re-running with a
different code state produces a different run_id.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from video_generation.store.models import CodeVersion

logger = logging.getLogger(__name__)

UNKNOWN_SHA = "unknown"


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.debug("git %s failed: %s", args, exc)
        return None
    return completed.stdout.strip()


def current_code_version(repo_root: Path | None = None) -> CodeVersion:
    """Return the current ``CodeVersion`` for ``repo_root`` (defaults to cwd).

    Falls back to ``CodeVersion("unknown", git_dirty=True)`` if git isn't
    available or the directory isn't a repo. The dirty flag is true on
    fallback so that runs from a non-repo are never confused with clean ones.
    """
    cwd = repo_root or Path.cwd()
    sha = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    if sha is None:
        return CodeVersion(git_sha=UNKNOWN_SHA, git_dirty=True)

    porcelain = _run_git(["status", "--porcelain"], cwd=cwd)
    dirty = bool(porcelain) if porcelain is not None else True
    return CodeVersion(git_sha=sha, git_dirty=dirty)
