"""Shared test fixtures."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from video_generation.store import (
    CodeVersion,
    ContentStore,
    LocalStorage,
    RunInputs,
    RunParams,
    RunStore,
)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture
def product_dir(tmp_path: Path) -> Path:
    """Create a temp directory with three dummy product images.

    Each image gets a slightly different fill colour so the three files have
    distinct sha256s (otherwise ``ContentStore.register_path`` would collapse
    them into a single content id and per-image steps like captioning would
    only run once).
    """
    d = tmp_path / "product"
    d.mkdir()
    fills = {
        "earring_on.webp": (200, 180, 160),
        "earring_front_look.webp": (210, 190, 170),
        "earring_side.png": (220, 200, 180),
    }
    for name, fill in fills.items():
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = fill
        ext = Path(name).suffix.lstrip(".")
        fmt = ".png" if ext == "webp" else f".{ext}"
        _, buf = cv2.imencode(fmt, img)
        (d / name).write_bytes(buf.tobytes())
    return d


@pytest.fixture
def analysis_file(tmp_path: Path) -> Path:
    """Create a dummy reference analysis markdown file."""
    p = tmp_path / "reference-analysis.md"
    p.write_text("# Reference Analysis\n\nA slow push-in shot of a model wearing jewelry.")
    return p


@pytest.fixture
def script_text() -> str:
    """Return a dummy script string."""
    return (
        "Single Continuous Shot: Diamond Star Earring Showcase\n\n"
        "Shot Description: A model with sleek blonde hair.\n"
        "Camera Movement: Slow push-in.\n"
    )


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """Create a minimal 3-frame synthetic video."""
    path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    for i in range(3):
        frame = np.full((48, 64, 3), fill_value=i * 80, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def dummy_frame_bytes() -> bytes:
    """Create a small PNG image and return its bytes."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (100, 150, 200)
    _, buf = cv2.imencode(".png", img)
    return bytes(buf.tobytes())


# ---------------------------------------------------------------------------
# Store-based fixtures (used by step tests that need a real RunStore)
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    return tmp_path / "store"


@pytest.fixture
def storage(storage_root: Path) -> LocalStorage:
    return LocalStorage(storage_root)


@pytest.fixture
def content_store(storage: LocalStorage) -> ContentStore:
    return ContentStore(storage)


@pytest.fixture
def run_store(storage: LocalStorage, content_store: ContentStore) -> RunStore:
    return RunStore(storage, content_store)


@pytest.fixture
def default_params() -> RunParams:
    return RunParams(
        claude_model="claude-test",
        gemini_model="gemini-test",
        edit_model="edit-test",
        video_model="video-test",
        video_duration="5",
        max_refinements=3,
    )


@pytest.fixture
def code_version() -> CodeVersion:
    return CodeVersion(git_sha="0" * 40, git_dirty=False)


@pytest.fixture
def empty_inputs() -> RunInputs:
    return RunInputs()
