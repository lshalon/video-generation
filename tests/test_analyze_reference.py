"""Tests for the analyze_reference step (Gemini-based)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from video_generation.steps.analyze_reference import OUTPUT_NAME, STEP_NAME, analyze_reference
from video_generation.store import (
    CodeVersion,
    ContentStore,
    RunInputs,
    RunParams,
    RunStore,
    StepContext,
)


def _make_ctx(
    run_store: RunStore,
    content_store: ContentStore,
    video_id: str,
    params: RunParams,
    code_version: CodeVersion,
) -> StepContext:
    inputs = RunInputs(reference_video=video_id)
    run = run_store.create_or_load(inputs=inputs, params=params, code_version=code_version)
    return StepContext(
        content_store, run_store, run.run_id, STEP_NAME, inputs={"reference_video": video_id}
    )


class TestAnalyzeReference:
    @patch("video_generation.steps.analyze_reference.get_gemini_client")
    def test_saves_analysis_and_returns_result(
        self,
        mock_get_client: MagicMock,
        tiny_video: Path,
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        video_ref = content_store.register_path(tiny_video, kind="video")
        ctx = _make_ctx(
            run_store, content_store, video_ref.content_id, default_params, code_version
        )

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_file = SimpleNamespace(name="files/abc123", state="ACTIVE")
        mock_client.files.upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "## Shot Breakdown\nGreat video."
        mock_client.models.generate_content.return_value = mock_response

        result = analyze_reference(
            video_ref.content_id,
            ctx=ctx,
            gemini_model="gemini-test-model",
            num_frames=2,
        )

        assert result.analysis_text == "## Shot Breakdown\nGreat video."
        assert result.content_id is not None
        assert result.output_path.exists()
        assert result.output_path.suffix == ".md"

        # The analysis was registered as content and attached to the step.
        run = run_store.get(ctx.run_id)
        assert run.steps[STEP_NAME].outputs[OUTPUT_NAME] == result.content_id

        mock_client.files.upload.assert_called_once()
        mock_client.models.generate_content.assert_called_once()
        assert mock_client.models.generate_content.call_args.kwargs["model"] == "gemini-test-model"

    @patch("video_generation.steps.analyze_reference.get_gemini_client")
    def test_polls_until_active(
        self,
        mock_get_client: MagicMock,
        tiny_video: Path,
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        video_ref = content_store.register_path(tiny_video, kind="video")
        ctx = _make_ctx(
            run_store, content_store, video_ref.content_id, default_params, code_version
        )

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_file_processing = SimpleNamespace(name="files/abc123", state="PROCESSING")
        mock_file_active = SimpleNamespace(name="files/abc123", state="ACTIVE")
        mock_client.files.upload.return_value = mock_file_processing
        mock_client.files.get.return_value = mock_file_active

        mock_response = MagicMock()
        mock_response.text = "Analysis text"
        mock_client.models.generate_content.return_value = mock_response

        with patch("video_generation.steps.analyze_reference.time") as mock_time:
            result = analyze_reference(video_ref.content_id, ctx=ctx)

        assert result.analysis_text == "Analysis text"
        mock_client.files.get.assert_called_once_with(name="files/abc123")
        mock_time.sleep.assert_called()

    @patch("video_generation.steps.analyze_reference.get_gemini_client")
    def test_cleans_up_uploaded_file(
        self,
        mock_get_client: MagicMock,
        tiny_video: Path,
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        video_ref = content_store.register_path(tiny_video, kind="video")
        ctx = _make_ctx(
            run_store, content_store, video_ref.content_id, default_params, code_version
        )

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.files.upload.return_value = SimpleNamespace(name="files/abc123", state="ACTIVE")
        mock_client.models.generate_content.return_value = MagicMock(text="Analysis")

        analyze_reference(video_ref.content_id, ctx=ctx)

        mock_client.files.delete.assert_called_once_with(name="files/abc123")
