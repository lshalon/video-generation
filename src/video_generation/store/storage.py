"""Storage backends for the content/run store.

A :class:`Storage` is a thin byte-level KV interface keyed by forward-slash
paths (e.g. ``library/ab/ab12...png``). Concrete backends include the local
filesystem (this module) and Google Drive (see ``gdrive.py``).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Byte-level key/value store backing the content + run layer."""

    def put(self, key: str, data: bytes, mime: str | None = None) -> None: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def list(self, prefix: str) -> list[str]: ...

    def delete(self, key: str) -> None: ...

    def url(self, key: str) -> str | None:
        """Return a backend-native URL for ``key`` if one exists.

        For :class:`LocalStorage` this is a ``file://`` URL; for cloud
        backends it might be a signed download URL. Returning ``None`` is
        valid for backends that have no notion of a URL.
        """


def _validate_key(key: str) -> str:
    """Reject absolute paths and parent-traversal segments."""
    if not key:
        raise ValueError("Storage key must be non-empty")
    if key.startswith("/"):
        raise ValueError(f"Storage key must be relative, got: {key!r}")
    parts = key.split("/")
    if any(part in ("", "..", ".") for part in parts):
        raise ValueError(f"Storage key has invalid segment: {key!r}")
    return key


class LocalStorage:
    """Filesystem-backed :class:`Storage` rooted at a single directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / _validate_key(key)

    def put(self, key: str, data: bytes, mime: str | None = None) -> None:
        del mime  # unused on local FS
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"Storage key not found: {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str) -> list[str]:
        prefix_path = self._path(prefix) if prefix else self.root
        if not prefix_path.exists():
            return []
        return sorted(_walk_keys(self.root, prefix_path))

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    def url(self, key: str) -> str | None:
        return self._path(key).as_uri()


def _walk_keys(root: Path, start: Path) -> Iterable[str]:
    """Yield storage keys (root-relative POSIX paths) for every file under ``start``."""
    if start.is_file():
        yield start.relative_to(root).as_posix()
        return
    for path in start.rglob("*"):
        if path.is_file():
            yield path.relative_to(root).as_posix()
