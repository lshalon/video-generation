"""Tests for the LocalStorage filesystem adapter."""

from pathlib import Path

import pytest

from video_generation.store import LocalStorage


class TestLocalStorageRoundTrip:
    def test_put_get(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.put("foo/bar.bin", b"hello")
        assert storage.get("foo/bar.bin") == b"hello"

    def test_exists(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        assert not storage.exists("missing/file.txt")
        storage.put("present.txt", b"x")
        assert storage.exists("present.txt")

    def test_delete(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.put("doomed.txt", b"bye")
        storage.delete("doomed.txt")
        assert not storage.exists("doomed.txt")

    def test_get_missing_raises(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        with pytest.raises(KeyError):
            storage.get("nope.txt")

    def test_list_returns_relative_keys(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.put("a/x.txt", b"1")
        storage.put("a/b/y.txt", b"2")
        storage.put("c/z.txt", b"3")
        assert storage.list("a") == ["a/b/y.txt", "a/x.txt"]
        assert "c/z.txt" in storage.list("")

    def test_list_missing_prefix(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        assert storage.list("nope") == []

    def test_url_returns_file_uri(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.put("a.bin", b"x")
        url = storage.url("a.bin")
        assert url is not None and url.startswith("file://")

    def test_atomic_overwrite(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.put("k.bin", b"first")
        storage.put("k.bin", b"second")
        assert storage.get("k.bin") == b"second"


class TestLocalStorageKeyValidation:
    @pytest.mark.parametrize("bad", ["", "/abs", "../escape", "a//b", "a/./b"])
    def test_rejects_bad_keys(self, tmp_path: Path, bad: str) -> None:
        storage = LocalStorage(tmp_path)
        with pytest.raises(ValueError):
            storage.put(bad, b"x")
