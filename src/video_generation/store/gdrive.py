"""Google Drive backed :class:`Storage` (stub).

This adapter mirrors the key namespace used by :class:`LocalStorage`: a key
like ``library/ab/ab12...png`` becomes a nested folder structure under the
configured root folder, with the file living at the leaf.

Authentication is via a service-account JSON credentials file. Set
``GOOGLE_DRIVE_CREDENTIALS`` to the path of the JSON file and pass the root
folder id either to :class:`GoogleDriveStorage` directly or via
``--storage gdrive:<folder-id>``.

This is intentionally a small, dependency-light wrapper. It is wired through
the :func:`video_generation.store.make_storage` factory and exercised by tests
that inject a fake Drive service, but it is not on the hot path for the local
default workflow.
"""

from __future__ import annotations

import builtins
import io
import logging
import os
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

logger = logging.getLogger(__name__)

FOLDER_MIME = "application/vnd.google-apps.folder"
DEFAULT_BINARY_MIME = "application/octet-stream"
DEFAULT_SCOPES = ("https://www.googleapis.com/auth/drive",)


def _build_default_service() -> Any:
    """Build a Drive v3 client from a service-account JSON credentials file.

    Reads ``GOOGLE_DRIVE_CREDENTIALS`` from the environment.
    """
    creds_path = os.environ.get("GOOGLE_DRIVE_CREDENTIALS")
    if not creds_path:
        raise RuntimeError(
            "GOOGLE_DRIVE_CREDENTIALS is not set. Point it at a service-account "
            "JSON file with Drive access to use the gdrive storage backend."
        )
    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=list(DEFAULT_SCOPES),
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


class GoogleDriveStorage:
    """Google Drive implementation of the :class:`Storage` interface."""

    def __init__(self, folder_id: str, service: Any = None) -> None:
        if not folder_id:
            raise ValueError("folder_id is required for GoogleDriveStorage")
        self.folder_id = folder_id
        self._service = service if service is not None else _build_default_service()
        self._folder_cache: dict[str, str] = {"": folder_id}

    # ------------------------------------------------------------------
    # Public Storage interface
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes, mime: str | None = None) -> None:
        parts = self._split(key)
        parent_id = self._ensure_folders(parts[:-1])
        leaf = parts[-1]
        existing_id = self._find_child(parent_id, leaf, FOLDER_MIME, negate=True)

        media = MediaIoBaseUpload(
            io.BytesIO(data),
            mimetype=mime or DEFAULT_BINARY_MIME,
            resumable=False,
        )

        if existing_id:
            self._service.files().update(fileId=existing_id, media_body=media).execute()
        else:
            self._service.files().create(
                body={"name": leaf, "parents": [parent_id]},
                media_body=media,
                fields="id",
            ).execute()

    def get(self, key: str) -> bytes:
        file_id = self._lookup(key)
        if file_id is None:
            raise KeyError(f"Storage key not found: {key}")

        request = self._service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()

    def exists(self, key: str) -> bool:
        return self._lookup(key) is not None

    def list(self, prefix: str) -> builtins.list[str]:
        parts: builtins.list[str] = self._split(prefix) if prefix else []
        try:
            start = self._lookup_folder(parts)
        except KeyError:
            return []
        if start is None:
            return []
        results: builtins.list[str] = []
        self._walk(start, parts, results)
        return sorted(results)

    def delete(self, key: str) -> None:
        file_id = self._lookup(key)
        if file_id is not None:
            self._service.files().delete(fileId=file_id).execute()

    def url(self, key: str) -> str | None:
        file_id = self._lookup(key)
        if file_id is None:
            return None
        return f"https://drive.google.com/file/d/{file_id}/view"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, key: str) -> builtins.list[str]:
        if not key:
            raise ValueError("Storage key must be non-empty")
        if key.startswith("/") or "//" in key:
            raise ValueError(f"Invalid storage key: {key!r}")
        parts = key.split("/")
        if any(p in ("", "..", ".") for p in parts):
            raise ValueError(f"Storage key has invalid segment: {key!r}")
        return parts

    def _ensure_folders(self, segments: builtins.list[str]) -> str:
        """Walk/create folders, return the deepest folder id."""
        path = ""
        parent_id = self.folder_id
        for segment in segments:
            path = f"{path}/{segment}" if path else segment
            cached = self._folder_cache.get(path)
            if cached:
                parent_id = cached
                continue
            existing = self._find_child(parent_id, segment, FOLDER_MIME)
            if existing is None:
                created = (
                    self._service.files()
                    .create(
                        body={
                            "name": segment,
                            "parents": [parent_id],
                            "mimeType": FOLDER_MIME,
                        },
                        fields="id",
                    )
                    .execute()
                )
                existing = created["id"]
            self._folder_cache[path] = existing
            parent_id = existing
        return parent_id

    def _lookup_folder(self, segments: builtins.list[str]) -> str | None:
        """Resolve a folder path to its Drive id (no creation)."""
        path = ""
        parent_id = self.folder_id
        for segment in segments:
            path = f"{path}/{segment}" if path else segment
            cached = self._folder_cache.get(path)
            if cached:
                parent_id = cached
                continue
            existing = self._find_child(parent_id, segment, FOLDER_MIME)
            if existing is None:
                return None
            self._folder_cache[path] = existing
            parent_id = existing
        return parent_id

    def _lookup(self, key: str) -> str | None:
        parts = self._split(key)
        parent_id = self._lookup_folder(parts[:-1])
        if parent_id is None:
            return None
        return self._find_child(parent_id, parts[-1], FOLDER_MIME, negate=True)

    def _find_child(
        self,
        parent_id: str,
        name: str,
        mime: str,
        negate: bool = False,
    ) -> str | None:
        """Find a single child of ``parent_id`` named ``name``.

        ``mime`` filters by mimeType; with ``negate=True`` it filters out
        ``mime`` (used to find non-folder children).
        """
        op = "!=" if negate else "="
        escaped_name = name.replace("\\", "\\\\").replace("'", "\\'")
        query = (
            f"name = '{escaped_name}' and "
            f"'{parent_id}' in parents and "
            f"mimeType {op} '{mime}' and "
            "trashed = false"
        )
        response = (
            self._service.files()
            .list(
                q=query,
                fields="files(id,name)",
                pageSize=1,
            )
            .execute()
        )
        files = response.get("files", [])
        return files[0]["id"] if files else None

    def _walk(self, folder_id: str, segments: builtins.list[str], out: builtins.list[str]) -> None:
        page_token: str | None = None
        while True:
            response = (
                self._service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id,name,mimeType)",
                    pageToken=page_token,
                )
                .execute()
            )
            for entry in response.get("files", []):
                child_path = [*segments, entry["name"]]
                if entry["mimeType"] == FOLDER_MIME:
                    self._walk(entry["id"], child_path, out)
                else:
                    out.append("/".join(child_path))
            page_token = response.get("nextPageToken")
            if not page_token:
                return
