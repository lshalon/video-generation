"""End-to-end pipeline test with all external clients monkey-patched.

Verifies:
- A full pipeline run produces a manifest with all four steps + content ids.
- The library contains blobs and metadata for inputs and outputs.
- Re-running with the same inputs is idempotent (same run_id, no extra
  upstream calls because cached step outputs short-circuit execution).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from video_generation.config import PipelineConfig
from video_generation.pipeline import run_pipeline
from video_generation.steps.generate_starting_frame import CompositesCritique


def _png_bytes() -> bytes:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return bytes(buf.tobytes())


def _accepted_critique() -> SimpleNamespace:
    payload = CompositesCritique(acceptable=True, issues=[], correction_prompt="")
    return SimpleNamespace(text=json.dumps(payload.model_dump()))


@pytest.fixture
def pipeline_config(product_dir: Path, analysis_file: Path, tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        product_dir=product_dir,
        reference_analysis=analysis_file,
        output_dir=tmp_path / "outputs",
        storage_spec=f"local:{tmp_path / 'store'}",
    )


@patch("video_generation.steps.generate_video.httpx")
@patch("video_generation.steps.generate_video.fal_client")
@patch("video_generation.steps.generate_video.get_anthropic_client")
@patch("video_generation.steps.generate_starting_frame.httpx")
@patch("video_generation.steps.generate_starting_frame.fal_client")
@patch("video_generation.steps.generate_starting_frame.get_gemini_client")
@patch("video_generation.steps.write_script.get_gemini_client")
@patch("video_generation.pipeline.load_env")
def test_full_pipeline_records_manifest_and_is_idempotent(
    _mock_load_env: MagicMock,
    mock_script_gemini: MagicMock,
    mock_frame_gemini: MagicMock,
    mock_frame_fal: MagicMock,
    mock_frame_httpx: MagicMock,
    mock_video_anthropic: MagicMock,
    mock_video_fal: MagicMock,
    mock_video_httpx: MagicMock,
    pipeline_config: PipelineConfig,
) -> None:
    # --- Stub Gemini (script step) ---
    script_client = MagicMock()
    mock_script_gemini.return_value = script_client
    script_client.models.generate_content.return_value = MagicMock(
        text="Slow push-in scene.",
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
    )

    # --- Stub Gemini (frame step: scene prompt + composite prompt + critique) ---
    gemini_client = MagicMock()
    mock_frame_gemini.return_value = gemini_client
    gemini_client.models.generate_content.side_effect = [
        MagicMock(text="Studio portrait, empty ears."),
        MagicMock(text="Composite prompt"),
        _accepted_critique(),
    ]

    # --- Stub fal + httpx for frame step ---
    fake_png = _png_bytes()
    mock_frame_fal.subscribe.side_effect = [
        {"images": [{"url": "https://fake/scene.png"}]},
        {"images": [{"url": "https://fake/composite.png"}]},
    ]
    mock_frame_fal.upload.return_value = "https://fake/upload.png"
    mock_frame_httpx.get.return_value = MagicMock(content=fake_png, raise_for_status=lambda: None)

    # --- Stub Claude (video step motion prompt) ---
    video_client = MagicMock()
    mock_video_anthropic.return_value = video_client
    video_client.messages.create.return_value = MagicMock(
        content=[SimpleNamespace(text="Slow push-in.")]
    )

    # --- Stub fal + httpx for video step ---
    mock_video_fal.upload.return_value = "https://fake/frame.jpg"
    mock_video_fal.subscribe.return_value = {"video": {"url": "https://fake/v.mp4"}}
    fake_mp4 = b"\x00\x00\x00\x18ftypmp42"
    mock_video_httpx.get.return_value = MagicMock(content=fake_mp4, raise_for_status=lambda: None)

    # --- First run ---
    result1 = run_pipeline(pipeline_config)

    # Manifest contains all 4 steps with completed status.
    assert result1.run_id is not None
    storage_root = Path(pipeline_config.storage_spec.removeprefix("local:"))
    manifest_path = storage_root / "runs" / result1.run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["steps"].keys()) == {"script", "frame", "video"}
    for step in manifest["steps"].values():
        assert step["status"] == "completed"

    # Frame step recorded its intermediates.
    frame_outputs = manifest["steps"]["frame"]["outputs"]
    assert "scene_prompt" in frame_outputs
    assert "composite_prompt" in frame_outputs
    assert "starting_frame_v0" in frame_outputs
    assert "starting_frame" in frame_outputs

    # Video step output exists on disk.
    assert result1.video_path.exists()
    assert result1.video_path.read_bytes() == fake_mp4

    # --- Second run (same config) is idempotent ---
    pre_call_count = (
        script_client.models.generate_content.call_count
        + gemini_client.models.generate_content.call_count
        + video_client.messages.create.call_count
    )
    result2 = run_pipeline(pipeline_config)

    assert result2.run_id == result1.run_id
    assert result2.video_content_id == result1.video_content_id
    post_call_count = (
        script_client.models.generate_content.call_count
        + gemini_client.models.generate_content.call_count
        + video_client.messages.create.call_count
    )
    # Cached steps should not have re-invoked any upstream API.
    assert post_call_count == pre_call_count
