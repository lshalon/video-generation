"""Tests for ContentStore (content-addressed blob registry)."""

import hashlib
import json
from pathlib import Path

import pytest

from video_generation.store import ContentStore, LocalStorage


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def store(tmp_path: Path) -> ContentStore:
    return ContentStore(LocalStorage(tmp_path))


class TestRegisterBytes:
    def test_returns_sha256(self, store: ContentStore) -> None:
        ref = store.register_bytes(b"hello", original_name="hi.txt")
        assert ref.content_id == _sha(b"hello")
        assert ref.meta.size == 5
        assert ref.meta.original_name == "hi.txt"
        assert ref.meta.kind == "text"

    def test_dedup_same_bytes(self, store: ContentStore) -> None:
        a = store.register_bytes(b"same", original_name="a.txt")
        b = store.register_bytes(b"same", original_name="b.txt")
        assert a.content_id == b.content_id
        # Original metadata wins; second registration is a no-op.
        assert b.meta.original_name == "a.txt"

    def test_get_bytes_round_trip(self, store: ContentStore) -> None:
        ref = store.register_bytes(
            b"payload", original_name="p.bin", mime="application/octet-stream"
        )
        assert store.get_bytes(ref.content_id) == b"payload"

    def test_meta_persisted_as_json(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        store = ContentStore(storage)
        ref = store.register_bytes(b"abc", original_name="x.json", mime="application/json")
        meta_key = f"library/{ref.content_id[:2]}/{ref.content_id}.meta.json"
        meta_data = json.loads(storage.get(meta_key))
        assert meta_data["sha256"] == ref.content_id
        assert meta_data["mime"] == "application/json"
        assert meta_data["kind"] == "json"

    def test_records_produced_by(self, store: ContentStore) -> None:
        ref = store.register_bytes(
            b"out",
            original_name="out.png",
            produced_by={"run_id": "r1", "step": "frame", "name": "starting_frame"},
        )
        assert store.get_meta(ref.content_id).produced_by == {
            "run_id": "r1",
            "step": "frame",
            "name": "starting_frame",
        }


class TestRegisterPath:
    def test_round_trip(self, store: ContentStore, tmp_path: Path) -> None:
        path = tmp_path / "input.txt"
        path.write_bytes(b"file bytes")
        ref = store.register_path(path)
        assert ref.meta.original_name == "input.txt"
        assert store.get_bytes(ref.content_id) == b"file bytes"

    def test_missing_raises(self, store: ContentStore, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            store.register_path(tmp_path / "missing.bin")


class TestMaterialize:
    def test_to_directory(self, store: ContentStore, tmp_path: Path) -> None:
        ref = store.register_bytes(b"data", original_name="src.png")
        dest_dir = tmp_path / "out"
        path = store.materialize(ref.content_id, dest_dir)
        assert path.exists()
        assert path.read_bytes() == b"data"
        assert path.suffix == ".png"

    def test_to_explicit_file(self, store: ContentStore, tmp_path: Path) -> None:
        ref = store.register_bytes(b"more", original_name="src.txt")
        target = tmp_path / "explicit.txt"
        path = store.materialize(ref.content_id, target)
        assert path == target
        assert path.read_bytes() == b"more"


class TestErrors:
    def test_get_unknown_id(self, store: ContentStore) -> None:
        with pytest.raises(KeyError):
            store.get_bytes("0" * 64)
