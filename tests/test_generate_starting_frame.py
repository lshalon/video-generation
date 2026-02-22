"""Tests for the generate_starting_frame step."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from video_generation.steps.generate_starting_frame import (
    CompositesCritique,
    generate_starting_frame,
)


def _fake_image_bytes() -> bytes:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return bytes(buf.tobytes())


def _mock_critique(acceptable: bool, issues: list[str], correction: str) -> MagicMock:
    """Create a mock response for messages.parse with a parsed_output."""
    critique = CompositesCritique(
        acceptable=acceptable, issues=issues, correction_prompt=correction
    )
    resp = MagicMock()
    resp.parsed_output = critique
    return resp


class TestGenerateStartingFrame:
    @patch("video_generation.steps.generate_starting_frame.httpx")
    @patch("video_generation.steps.generate_starting_frame.fal_client")
    @patch("video_generation.steps.generate_starting_frame.get_anthropic_client")
    def test_accepted_on_first_try(
        self,
        mock_get_client: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        product_dir: Path,
        tmp_output_dir: Path,
        script_text: str,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # messages.create: scene prompt, composite prompt
        mock_client.messages.create.side_effect = [
            MagicMock(content=[SimpleNamespace(text="Studio portrait, empty ears.")]),
            MagicMock(content=[SimpleNamespace(text="Place the product on the ear.")]),
        ]
        # messages.parse: critique (accepted)
        mock_client.messages.parse.side_effect = [
            _mock_critique(True, [], ""),
        ]

        fake_bytes = _fake_image_bytes()
        mock_fal.subscribe.side_effect = [
            {"images": [{"url": "https://fake/scene.png"}]},
            {"images": [{"url": "https://fake/composite.png"}]},
        ]
        mock_fal.upload.return_value = "https://fake/uploaded.png"

        mock_resp = MagicMock()
        mock_resp.content = fake_bytes
        mock_httpx.get.return_value = mock_resp

        result = generate_starting_frame(
            script_text,
            product_dir,
            claude_model="test-model",
            max_refinements=3,
            output_dir=tmp_output_dir,
        )

        assert result.frame_path.exists()
        assert result.frame_path.suffix == ".png"

        assert mock_client.messages.create.call_count == 2
        assert mock_client.messages.parse.call_count == 1
        assert mock_fal.subscribe.call_count == 2

    @patch("video_generation.steps.generate_starting_frame.httpx")
    @patch("video_generation.steps.generate_starting_frame.fal_client")
    @patch("video_generation.steps.generate_starting_frame.get_anthropic_client")
    def test_refines_once_then_accepted(
        self,
        mock_get_client: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        product_dir: Path,
        tmp_output_dir: Path,
        script_text: str,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.messages.create.side_effect = [
            MagicMock(content=[SimpleNamespace(text="Studio portrait.")]),
            MagicMock(content=[SimpleNamespace(text="Place product on ear.")]),
        ]
        mock_client.messages.parse.side_effect = [
            _mock_critique(False, ["Product is too large"], "Make the product 50% smaller."),
            _mock_critique(True, [], ""),
        ]

        fake_bytes = _fake_image_bytes()
        mock_fal.subscribe.side_effect = [
            {"images": [{"url": "https://fake/scene.png"}]},
            {"images": [{"url": "https://fake/v0.png"}]},
            {"images": [{"url": "https://fake/v1.png"}]},
        ]
        mock_fal.upload.return_value = "https://fake/uploaded.png"

        mock_resp = MagicMock()
        mock_resp.content = fake_bytes
        mock_httpx.get.return_value = mock_resp

        result = generate_starting_frame(
            script_text,
            product_dir,
            claude_model="test-model",
            max_refinements=3,
            output_dir=tmp_output_dir,
        )

        assert result.frame_path.exists()

        assert mock_client.messages.create.call_count == 2
        assert mock_client.messages.parse.call_count == 2
        assert mock_fal.subscribe.call_count == 3

        assert (tmp_output_dir / "critique_v1.json").exists()
        assert (tmp_output_dir / "critique_v2.json").exists()

    @patch("video_generation.steps.generate_starting_frame.httpx")
    @patch("video_generation.steps.generate_starting_frame.fal_client")
    @patch("video_generation.steps.generate_starting_frame.get_anthropic_client")
    def test_stops_at_max_refinements(
        self,
        mock_get_client: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        product_dir: Path,
        tmp_output_dir: Path,
        script_text: str,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.messages.create.side_effect = [
            MagicMock(content=[SimpleNamespace(text="Studio portrait.")]),
            MagicMock(content=[SimpleNamespace(text="Place product.")]),
        ]
        mock_client.messages.parse.side_effect = [
            _mock_critique(False, ["Still too large"], "Make it even smaller."),
            _mock_critique(False, ["Still too large"], "Make it even smaller."),
        ]

        fake_bytes = _fake_image_bytes()
        mock_fal.subscribe.side_effect = [
            {"images": [{"url": "https://fake/scene.png"}]},
            {"images": [{"url": "https://fake/v0.png"}]},
            {"images": [{"url": "https://fake/v1.png"}]},
            {"images": [{"url": "https://fake/v2.png"}]},
        ]
        mock_fal.upload.return_value = "https://fake/uploaded.png"

        mock_resp = MagicMock()
        mock_resp.content = fake_bytes
        mock_httpx.get.return_value = mock_resp

        result = generate_starting_frame(
            script_text,
            product_dir,
            claude_model="test-model",
            max_refinements=2,
            output_dir=tmp_output_dir,
        )

        assert result.frame_path.exists()
