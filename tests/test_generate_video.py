"""Tests for the generate_video step."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from video_generation.steps.generate_video import (
    OUTPUT_NAME,
    STEP_NAME,
    _build_video_arguments,
    generate_video,
)
from video_generation.store import (
    CodeVersion,
    ContentStore,
    RunInputs,
    RunParams,
    RunStore,
    StepContext,
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


def _setup(
    content_store: ContentStore,
    run_store: RunStore,
    frame_bytes: bytes,
    script_text: str,
    params: RunParams,
    code_version: CodeVersion,
) -> tuple[StepContext, str, str]:
    frame_ref = content_store.register_bytes(
        frame_bytes, original_name="starting_frame.png", kind="image"
    )
    script_ref = content_store.register_bytes(
        script_text.encode("utf-8"), original_name="script.md", kind="text"
    )
    inputs = RunInputs(reference_analysis=script_ref.content_id)
    run = run_store.create_or_load(inputs=inputs, params=params, code_version=code_version)
    ctx = StepContext(
        content_store,
        run_store,
        run.run_id,
        STEP_NAME,
        inputs={"starting_frame": frame_ref.content_id, "script": script_ref.content_id},
    )
    return ctx, frame_ref.content_id, script_ref.content_id


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
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        ctx, frame_id, script_id = _setup(
            content_store, run_store, dummy_frame_bytes, script_text, default_params, code_version
        )

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [SimpleNamespace(text="Slow push-in with head tilt.")]
        mock_client.messages.create.return_value = mock_response

        mock_fal.upload.return_value = "https://fal.media/uploaded.jpg"
        mock_fal.subscribe.return_value = {"video": {"url": "https://fal.media/video.mp4"}}

        fake_video_bytes = b"\x00\x00\x00\x18ftypmp42"
        mock_video_resp = MagicMock()
        mock_video_resp.content = fake_video_bytes
        mock_httpx.get.return_value = mock_video_resp

        result = generate_video(
            frame_id,
            script_id,
            ctx=ctx,
            claude_model="test-model",
            video_model="fal-ai/kling-video/v2.6/pro/image-to-video",
            duration="5",
        )

        assert result.content_id is not None
        assert result.video_path.exists()
        assert result.video_path.suffix == ".mp4"

        run = run_store.get(ctx.run_id)
        outputs = run.steps[STEP_NAME].outputs
        assert outputs[OUTPUT_NAME] == result.content_id
        assert "motion_prompt" in outputs

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
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        ctx, frame_id, script_id = _setup(
            content_store, run_store, dummy_frame_bytes, script_text, default_params, code_version
        )

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
            frame_id,
            script_id,
            ctx=ctx,
            claude_model="test-model",
            video_model="fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
            duration="8",
        )

        assert result.content_id is not None

        subscribe_args = mock_fal.subscribe.call_args
        api_args = subscribe_args.kwargs["arguments"]
        assert "image_url" in api_args
        assert "start_image_url" not in api_args
        assert api_args["duration"] == "8"
