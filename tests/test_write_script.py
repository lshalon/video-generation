"""Tests for the write_script step."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from video_generation.steps.write_script import (
    OUTPUT_NAME,
    STEP_NAME,
    discover_product_images,
    write_script,
)
from video_generation.store import (
    CodeVersion,
    ContentStore,
    RunInputs,
    RunParams,
    RunStore,
    StepContext,
)


class TestDiscoverProductImages:
    def test_finds_images(self, product_dir: Path) -> None:
        images = discover_product_images(product_dir)
        assert len(images) == 3
        names = {p.name for p in images}
        assert "earring_on.webp" in names
        assert "earring_side.png" in names

    def test_ignores_hidden_files(self, product_dir: Path) -> None:
        (product_dir / ".DS_Store").write_text("junk")
        images = discover_product_images(product_dir)
        assert all(not p.name.startswith(".") for p in images)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No product images"):
            discover_product_images(empty)

    def test_sorted_by_name(self, product_dir: Path) -> None:
        names = [p.name for p in discover_product_images(product_dir)]
        assert names == sorted(names)


def _make_ctx(
    run_store: RunStore,
    content_store: ContentStore,
    analysis_id: str,
    product_ids: list[str],
    params: RunParams,
    code_version: CodeVersion,
) -> StepContext:
    inputs = RunInputs(reference_analysis=analysis_id, product_images=list(product_ids))
    run = run_store.create_or_load(inputs=inputs, params=params, code_version=code_version)
    return StepContext(
        content_store,
        run_store,
        run.run_id,
        STEP_NAME,
        inputs={"reference_analysis": analysis_id, "product_images": list(product_ids)},
    )


class TestWriteScript:
    @patch("video_generation.steps.write_script.get_gemini_client")
    def test_generates_and_saves_script(
        self,
        mock_get_client: MagicMock,
        analysis_file: Path,
        product_dir: Path,
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        analysis_ref = content_store.register_path(analysis_file, kind="text")
        product_refs = [
            content_store.register_path(p, kind="image")
            for p in discover_product_images(product_dir)
        ]
        product_ids = [r.content_id for r in product_refs]

        ctx = _make_ctx(
            run_store,
            content_store,
            analysis_ref.content_id,
            product_ids,
            default_params,
            code_version,
        )

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "**Single Shot**: Camera pushes in slowly."
        mock_response.usage_metadata = SimpleNamespace(
            prompt_token_count=200, candidates_token_count=100
        )
        mock_client.models.generate_content.return_value = mock_response

        result = write_script(
            analysis_ref.content_id,
            product_ids,
            ctx=ctx,
            gemini_model="test-model",
        )

        assert result.script_text == "**Single Shot**: Camera pushes in slowly."
        assert result.content_id is not None
        assert result.script_path.exists()
        assert result.script_path.name == "script.md"

        run = run_store.get(ctx.run_id)
        assert run.steps[STEP_NAME].outputs[OUTPUT_NAME] == result.content_id
        assert run.steps[STEP_NAME].attributes["gemini_model"] == "test-model"

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        contents = call_kwargs["contents"]
        image_parts = [c for c in contents if hasattr(c, "inline_data")]
        assert len(image_parts) == 3
        text_parts = [c for c in contents if isinstance(c, str)]
        assert len(text_parts) == 1
        assert "product images I have available" in text_parts[0]
