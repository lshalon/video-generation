"""Tests for the analyze_reference step (Gemini-based)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from video_generation.steps.analyze_reference import analyze_reference


class TestAnalyzeReference:
    @patch("video_generation.steps.analyze_reference.get_gemini_client")
    def test_saves_analysis_and_returns_result(
        self, mock_get_client: MagicMock, tiny_video: Path, tmp_output_dir: Path
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock file upload → immediately ACTIVE
        mock_file = SimpleNamespace(name="files/abc123", state="ACTIVE")
        mock_client.files.upload.return_value = mock_file

        # Mock generate_content response
        mock_response = MagicMock()
        mock_response.text = "## Shot Breakdown\nGreat video."
        mock_client.models.generate_content.return_value = mock_response

        result = analyze_reference(
            tiny_video,
            gemini_model="gemini-test-model",
            num_frames=2,
            output_dir=tmp_output_dir,
        )

        assert result.output_path.exists()
        assert result.analysis_text == "## Shot Breakdown\nGreat video."
        assert result.output_path.suffix == ".md"

        mock_client.files.upload.assert_called_once()
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-test-model"

    @patch("video_generation.steps.analyze_reference.get_gemini_client")
    def test_polls_until_active(
        self, mock_get_client: MagicMock, tiny_video: Path, tmp_output_dir: Path
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Upload returns PROCESSING, then get() returns ACTIVE
        mock_file_processing = SimpleNamespace(name="files/abc123", state="PROCESSING")
        mock_file_active = SimpleNamespace(name="files/abc123", state="ACTIVE")
        mock_client.files.upload.return_value = mock_file_processing
        mock_client.files.get.return_value = mock_file_active

        mock_response = MagicMock()
        mock_response.text = "Analysis text"
        mock_client.models.generate_content.return_value = mock_response

        with patch("video_generation.steps.analyze_reference.time") as mock_time:
            result = analyze_reference(
                tiny_video,
                output_dir=tmp_output_dir,
            )

        assert result.analysis_text == "Analysis text"
        mock_client.files.get.assert_called_once_with(name="files/abc123")
        mock_time.sleep.assert_called()

    @patch("video_generation.steps.analyze_reference.get_gemini_client")
    def test_cleans_up_uploaded_file(
        self, mock_get_client: MagicMock, tiny_video: Path, tmp_output_dir: Path
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_file = SimpleNamespace(name="files/abc123", state="ACTIVE")
        mock_client.files.upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Analysis"
        mock_client.models.generate_content.return_value = mock_response

        analyze_reference(tiny_video, output_dir=tmp_output_dir)

        mock_client.files.delete.assert_called_once_with(name="files/abc123")

    @patch("video_generation.steps.analyze_reference.get_gemini_client")
    def test_defaults_output_to_video_parent(
        self, mock_get_client: MagicMock, tiny_video: Path
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_file = SimpleNamespace(name="files/abc123", state="ACTIVE")
        mock_client.files.upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Analysis"
        mock_client.models.generate_content.return_value = mock_response

        result = analyze_reference(tiny_video)

        expected = tiny_video.parent / f"{tiny_video.stem}-analysis.md"
        assert result.output_path == expected
