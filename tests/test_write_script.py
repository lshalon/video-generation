"""Tests for the write_script step."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from video_generation.steps.write_script import (
    load_product_images,
    write_script,
)


class TestLoadProductImages:
    def test_finds_images(self, product_dir: Path) -> None:
        images = load_product_images(product_dir)
        assert len(images) == 3
        names = {p.name for p in images}
        assert "earring_on.webp" in names
        assert "earring_side.png" in names

    def test_ignores_hidden_files(self, product_dir: Path) -> None:
        (product_dir / ".DS_Store").write_text("junk")
        images = load_product_images(product_dir)
        assert all(not p.name.startswith(".") for p in images)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No product images"):
            load_product_images(empty)

    def test_sorted_by_name(self, product_dir: Path) -> None:
        images = load_product_images(product_dir)
        names = [p.name for p in images]
        assert names == sorted(names)


class TestWriteScript:
    @patch("video_generation.steps.write_script.get_anthropic_client")
    def test_generates_and_saves_script(
        self,
        mock_get_client: MagicMock,
        analysis_file: Path,
        product_dir: Path,
        tmp_output_dir: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="**Single Shot**: Camera pushes in slowly.")]
        mock_response.usage = SimpleNamespace(input_tokens=200, output_tokens=100)
        mock_client.messages.create.return_value = mock_response

        result = write_script(
            analysis_file,
            product_dir,
            claude_model="test-model",
            output_dir=tmp_output_dir,
        )

        assert result.script_path.exists()
        assert result.script_text == "**Single Shot**: Camera pushes in slowly."
        assert result.script_path.name == "script.md"

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"
        content = call_kwargs.kwargs["messages"][0]["content"]
        image_blocks = [c for c in content if c["type"] == "image"]
        assert len(image_blocks) == 3  # 3 product images
