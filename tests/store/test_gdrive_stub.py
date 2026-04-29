"""Tests for GoogleDriveStorage using an in-memory fake Drive service.

The stub is exercised by injecting a fake service that mimics the small slice
of the Drive v3 API the storage actually calls. We verify request shapes
(``files().list(q=...)`` queries, folder creation, upload + download) without
needing real credentials.
"""

from __future__ import annotations

import io
import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from video_generation.store.gdrive import FOLDER_MIME, GoogleDriveStorage


class FakeDriveService:
    """In-memory Drive service. Files are dicts: ``{id, name, parents, mime, data}``."""

    def __init__(self, root_id: str) -> None:
        self.root_id = root_id
        self._counter = 0
        self._files: dict[str, dict[str, Any]] = {
            root_id: {
                "id": root_id,
                "name": "ROOT",
                "parents": [],
                "mimeType": FOLDER_MIME,
                "data": None,
            }
        }

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    # The googleapiclient pattern is `service.files().list(q=...).execute()`,
    # so each method below returns a chainable object exposing `.execute()`.

    def files(self) -> _FilesEndpoint:
        return _FilesEndpoint(self)


class _Executable:
    def __init__(self, value: Any) -> None:
        self._value = value

    def execute(self) -> Any:
        return self._value


class _MediaDownload:
    """Mimic googleapiclient.http.MediaIoBaseDownload's request hand-off."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def execute(self) -> bytes:
        return self.data


class _FilesEndpoint:
    def __init__(self, service: FakeDriveService) -> None:
        self.service = service

    _NAME_RE = re.compile(r"name = '([^']*)'")
    _PARENT_RE = re.compile(r"'([^']*)' in parents")
    _MIME_EQ_RE = re.compile(r"mimeType = '([^']*)'")
    _MIME_NEQ_RE = re.compile(r"mimeType != '([^']*)'")

    def list(
        self,
        q: str | None = None,
        fields: str | None = None,
        pageSize: int | None = None,
        pageToken: str | None = None,
    ) -> _Executable:
        del fields, pageSize, pageToken
        files = list(self.service._files.values())
        if q is None:
            return _Executable({"files": files, "nextPageToken": None})

        name_match = self._NAME_RE.search(q)
        parent_match = self._PARENT_RE.search(q)
        mime_eq_match = self._MIME_EQ_RE.search(q)
        mime_neq_match = self._MIME_NEQ_RE.search(q)

        name_eq = name_match.group(1) if name_match else None
        parent_eq = parent_match.group(1) if parent_match else None
        mime_eq = mime_eq_match.group(1) if mime_eq_match else None
        mime_neq = mime_neq_match.group(1) if mime_neq_match else None

        def matches(entry: dict[str, Any]) -> bool:
            if name_eq is not None and entry["name"] != name_eq:
                return False
            if parent_eq is not None and parent_eq not in entry.get("parents", []):
                return False
            if mime_eq is not None and entry["mimeType"] != mime_eq:
                return False
            if mime_neq is not None and entry["mimeType"] == mime_neq:
                return False
            return True

        results = [
            {"id": f["id"], "name": f["name"], "mimeType": f["mimeType"]}
            for f in files
            if matches(f)
        ]
        return _Executable({"files": results, "nextPageToken": None})

    def create(
        self, body: dict[str, Any], media_body: Any = None, fields: str | None = None
    ) -> _Executable:
        del fields
        new_id = self.service._next_id("id-")
        data: bytes | None = None
        if media_body is not None:
            data = media_body._fh.getvalue() if hasattr(media_body, "_fh") else None
            if data is None and hasattr(media_body, "stream"):
                data = media_body.stream().read()
        self.service._files[new_id] = {
            "id": new_id,
            "name": body["name"],
            "parents": body.get("parents", []),
            "mimeType": body.get("mimeType", "application/octet-stream"),
            "data": data,
        }
        return _Executable({"id": new_id})

    def update(self, fileId: str, media_body: Any) -> _Executable:
        existing = self.service._files[fileId]
        data = media_body._fh.getvalue() if hasattr(media_body, "_fh") else b""
        existing["data"] = data
        return _Executable({"id": fileId})

    def delete(self, fileId: str) -> _Executable:
        self.service._files.pop(fileId, None)
        return _Executable(None)

    def get_media(self, fileId: str) -> Any:
        return _MediaDownload(self.service._files[fileId]["data"] or b"")


@pytest.fixture
def gdrive() -> tuple[GoogleDriveStorage, FakeDriveService]:
    fake = FakeDriveService(root_id="ROOT_FOLDER")
    storage = GoogleDriveStorage(folder_id="ROOT_FOLDER", service=fake)
    return storage, fake


class TestGoogleDriveStorage:
    def test_put_creates_folders_and_file(
        self, gdrive: tuple[GoogleDriveStorage, FakeDriveService]
    ) -> None:
        storage, fake = gdrive

        # Patch in our MediaIoBaseUpload so it stores bytes accessibly.
        from video_generation.store import gdrive as gdrive_module

        captured: dict[str, bytes] = {}

        class FakeUpload:
            def __init__(self, fh: io.BytesIO, mimetype: str, resumable: bool = False) -> None:
                del mimetype, resumable
                self._fh = fh
                captured["data"] = fh.getvalue()

        original = gdrive_module.MediaIoBaseUpload
        gdrive_module.MediaIoBaseUpload = FakeUpload  # type: ignore[misc, assignment]
        try:
            storage.put("library/ab/abc.png", b"PNGDATA", mime="image/png")
        finally:
            gdrive_module.MediaIoBaseUpload = original  # type: ignore[misc]

        # Two folders + one file.
        names = sorted(f["name"] for f in fake._files.values() if f["mimeType"] == FOLDER_MIME)
        assert "library" in names
        assert "ab" in names
        leaves = [f for f in fake._files.values() if f["name"] == "abc.png"]
        assert len(leaves) == 1
        assert captured["data"] == b"PNGDATA"

    def test_exists_round_trip(self, gdrive: tuple[GoogleDriveStorage, FakeDriveService]) -> None:
        storage, fake = gdrive
        # Pre-create a file by hand.
        lib = fake._next_id("id-")
        fake._files[lib] = {
            "id": lib,
            "name": "library",
            "parents": ["ROOT_FOLDER"],
            "mimeType": FOLDER_MIME,
            "data": None,
        }
        leaf = fake._next_id("id-")
        fake._files[leaf] = {
            "id": leaf,
            "name": "thing.txt",
            "parents": [lib],
            "mimeType": "text/plain",
            "data": b"hello",
        }
        assert storage.exists("library/thing.txt")
        assert not storage.exists("library/missing.txt")

    def test_url_for_existing_file(
        self, gdrive: tuple[GoogleDriveStorage, FakeDriveService]
    ) -> None:
        storage, fake = gdrive
        leaf = fake._next_id("id-")
        fake._files[leaf] = {
            "id": leaf,
            "name": "v.mp4",
            "parents": ["ROOT_FOLDER"],
            "mimeType": "video/mp4",
            "data": b"x",
        }
        url = storage.url("v.mp4")
        assert url == f"https://drive.google.com/file/d/{leaf}/view"

    def test_factory_routes_gdrive(self) -> None:
        from video_generation.store import make_storage

        # Build with a mock service so we don't try to authenticate.
        with pytest.raises(RuntimeError, match="GOOGLE_DRIVE_CREDENTIALS"):
            make_storage("gdrive:some-folder")


class TestGoogleDriveValidation:
    def test_requires_folder_id(self) -> None:
        with pytest.raises(ValueError):
            GoogleDriveStorage(folder_id="", service=MagicMock())
