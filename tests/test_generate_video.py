"""Tests for the generate_video step."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from video_generation.steps.generate_video import (
    _build_video_arguments,
    generate_video,
)


class TestBuildVideoArguments:
    def test_kling_arguments(self) -> None:
        args = _build_video_arguments("slow push-in", "https://img.url", "5", is_seedance=False)
        assert "start_image_url" in args
        assert "image_url" not in args
        assert args["negative_prompt"] == "blur, distort, low quality, jitter, shake"
        assert args["duration"] == "5"

    def test_seedance_arguments(self) -> None:
        args = _build_video_arguments("slow push-in", "https://img.url", "8", is_seedance=True)
        assert "image_url" in args
        assert "start_image_url" not in args
        assert "negative_prompt" not in args
        assert args["duration"] == "8"


class TestGenerateVideo:
    @patch("video_generation.steps.generate_video.httpx")
    @patch("video_generation.steps.generate_video.fal_client")
    @patch("video_generation.steps.generate_video.get_anthropic_client")
    def test_kling_pipeline(
        self,
        mock_get_client: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        dummy_frame_bytes: bytes,
        script_text: str,
        tmp_path: Path,
    ) -> None:
        frame_path = tmp_path / "frame.png"
        frame_path.write_bytes(dummy_frame_bytes)
        out_dir = tmp_path / "videos"
        out_dir.mkdir()

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="Slow push-in with head tilt.")]
        mock_client.messages.create.return_value = mock_response

        mock_fal.upload.return_value = "https://fal.media/uploaded.jpg"
        mock_fal.subscribe.return_value = {"video": {"url": "https://fal.media/video.mp4"}}

        fake_video_bytes = b"\x00\x00\x00\x18ftypmp42"  # minimal mp4-ish header
        mock_video_resp = MagicMock()
        mock_video_resp.content = fake_video_bytes
        mock_httpx.get.return_value = mock_video_resp

        result = generate_video(
            frame_path,
            script_text,
            claude_model="test-model",
            video_model="fal-ai/kling-video/v2.6/pro/image-to-video",
            duration="5",
            output_dir=out_dir,
        )

        assert result.video_path.exists()
        assert result.video_path.suffix == ".mp4"

        subscribe_args = mock_fal.subscribe.call_args
        assert subscribe_args.args[0] == "fal-ai/kling-video/v2.6/pro/image-to-video"
        api_args = subscribe_args.kwargs["arguments"]
        assert "start_image_url" in api_args
        assert "image_url" not in api_args

    @patch("video_generation.steps.generate_video.httpx")
    @patch("video_generation.steps.generate_video.fal_client")
    @patch("video_generation.steps.generate_video.get_anthropic_client")
    def test_seedance_pipeline(
        self,
        mock_get_client: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        dummy_frame_bytes: bytes,
        script_text: str,
        tmp_path: Path,
    ) -> None:
        frame_path = tmp_path / "frame.png"
        frame_path.write_bytes(dummy_frame_bytes)
        out_dir = tmp_path / "videos"
        out_dir.mkdir()

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="Slow zoom.")]
        mock_client.messages.create.return_value = mock_response

        mock_fal.upload.return_value = "https://fal.media/uploaded.jpg"
        mock_fal.subscribe.return_value = {"video": {"url": "https://fal.media/video.mp4"}}

        mock_video_resp = MagicMock()
        mock_video_resp.content = b"\x00\x00\x00\x18ftypmp42"
        mock_httpx.get.return_value = mock_video_resp

        result = generate_video(
            frame_path,
            script_text,
            claude_model="test-model",
            video_model="fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
            duration="8",
            output_dir=out_dir,
        )

        assert result.video_path.exists()

        subscribe_args = mock_fal.subscribe.call_args
        api_args = subscribe_args.kwargs["arguments"]
        assert "image_url" in api_args
        assert "start_image_url" not in api_args
        assert api_args["duration"] == "8"
