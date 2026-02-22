"""Shared test fixtures."""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture
def product_dir(tmp_path: Path) -> Path:
    """Create a temp directory with dummy product images."""
    d = tmp_path / "product"
    d.mkdir()
    for name in ["earring_on.webp", "earring_front_look.webp", "earring_side.png"]:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (200, 180, 160)  # light tan
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
