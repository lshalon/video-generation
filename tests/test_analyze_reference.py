"""Tests for the analyze_reference step."""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from video_generation.steps.analyze_reference import (
    analyze_reference,
    extract_frames,
    frame_to_base64,
)


class TestExtractFrames:
    def test_extracts_correct_count(self, tiny_video: Path) -> None:
        frames = extract_frames(tiny_video, num_frames=2)
        assert len(frames) == 2
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_extracts_all_frames(self, tiny_video: Path) -> None:
        frames = extract_frames(tiny_video, num_frames=3)
        assert len(frames) == 3

    def test_invalid_video_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.mp4"
        bad.write_text("not a video")
        with pytest.raises(ValueError, match="Could not open video"):
            extract_frames(bad, num_frames=1)


class TestFrameToBase64:
    def test_returns_valid_base64(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = frame_to_base64(frame)
        decoded = base64.standard_b64decode(result)
        assert len(decoded) > 0

    def test_respects_max_size(self) -> None:
        frame = np.zeros((2000, 3000, 3), dtype=np.uint8)
        result = frame_to_base64(frame, max_size=512)
        decoded = base64.standard_b64decode(result)
        assert len(decoded) > 0


class TestAnalyzeReference:
    @patch("video_generation.steps.analyze_reference.get_anthropic_client")
    def test_saves_analysis_and_returns_result(
        self, mock_get_client: MagicMock, tiny_video: Path, tmp_output_dir: Path
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="## Shot Breakdown\nGreat video.")]
        mock_response.usage = SimpleNamespace(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response

        result = analyze_reference(
            tiny_video,
            claude_model="test-model",
            num_frames=2,
            output_dir=tmp_output_dir,
        )

        assert result.output_path.exists()
        assert result.analysis_text == "## Shot Breakdown\nGreat video."
        assert result.output_path.suffix == ".md"

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"
