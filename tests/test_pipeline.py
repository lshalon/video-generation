"""Tests for the pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_generation.config import (
    AnalysisResult,
    PipelineConfig,
    ScriptResult,
    StartingFrameResult,
    VideoResult,
)
from video_generation.pipeline import run_pipeline, run_step


def _config(
    product_dir: Path,
    storage_root: Path,
    reference_video: Path | None = None,
    reference_analysis: Path | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        product_dir=product_dir,
        reference_video=reference_video,
        reference_analysis=reference_analysis,
        output_dir=storage_root / "outputs",
        storage_spec=f"local:{storage_root}",
    )


class TestRunPipeline:
    @patch("video_generation.pipeline.generate_video")
    @patch("video_generation.pipeline.generate_starting_frame")
    @patch("video_generation.pipeline.write_script")
    @patch("video_generation.pipeline.load_env")
    def test_full_pipeline_with_existing_analysis(
        self,
        _mock_load_env: MagicMock,
        mock_write_script: MagicMock,
        mock_gen_frame: MagicMock,
        mock_gen_video: MagicMock,
        product_dir: Path,
        analysis_file: Path,
        storage_root: Path,
    ) -> None:
        cfg = _config(product_dir, storage_root, reference_analysis=analysis_file)

        mock_write_script.return_value = ScriptResult(
            script_text="Test script",
            script_path=storage_root / "script.md",
            content_id="script-id",
        )
        mock_gen_frame.return_value = StartingFrameResult(
            frame_path=storage_root / "frame.png",
            content_id="frame-id",
        )
        mock_gen_video.return_value = VideoResult(
            video_path=storage_root / "video.mp4",
            content_id="video-id",
        )

        result = run_pipeline(cfg)

        assert result.video_content_id == "video-id"
        assert result.script_content_id == "script-id"
        assert result.starting_frame_content_id == "frame-id"
        assert result.run_id is not None

        # write_script received the analysis content_id (not a path)
        assert mock_write_script.call_args.args[0] is not None
        # frame received the script content_id
        assert mock_gen_frame.call_args.args[0] == "script-id"
        # video received frame_id, script_id
        assert mock_gen_video.call_args.args[0] == "frame-id"
        assert mock_gen_video.call_args.args[1] == "script-id"

    @patch("video_generation.pipeline.generate_video")
    @patch("video_generation.pipeline.generate_starting_frame")
    @patch("video_generation.pipeline.write_script")
    @patch("video_generation.pipeline.analyze_reference")
    @patch("video_generation.pipeline.load_env")
    def test_full_pipeline_with_video_runs_analysis(
        self,
        _mock_load_env: MagicMock,
        mock_analyze: MagicMock,
        mock_write_script: MagicMock,
        mock_gen_frame: MagicMock,
        mock_gen_video: MagicMock,
        product_dir: Path,
        tiny_video: Path,
        storage_root: Path,
    ) -> None:
        cfg = _config(product_dir, storage_root, reference_video=tiny_video)

        mock_analyze.return_value = AnalysisResult(
            analysis_text="analysis", output_path=storage_root / "a.md", content_id="analysis-id"
        )
        mock_write_script.return_value = ScriptResult(
            script_text="script", script_path=storage_root / "s.md", content_id="script-id"
        )
        mock_gen_frame.return_value = StartingFrameResult(
            frame_path=storage_root / "f.png", content_id="frame-id"
        )
        mock_gen_video.return_value = VideoResult(
            video_path=storage_root / "v.mp4", content_id="video-id"
        )

        run_pipeline(cfg)

        mock_analyze.assert_called_once()
        mock_write_script.assert_called_once()
        # script step receives the analysis content id from analyze
        assert mock_write_script.call_args.args[0] == "analysis-id"

    @patch("video_generation.pipeline.load_env")
    def test_missing_references_raises(
        self, _mock_load_env: MagicMock, product_dir: Path, storage_root: Path
    ) -> None:
        cfg = PipelineConfig(
            product_dir=product_dir,
            output_dir=storage_root / "outputs",
            storage_spec=f"local:{storage_root}",
        )
        with pytest.raises(ValueError):
            run_pipeline(cfg)


class TestRunStep:
    @patch("video_generation.pipeline.analyze_reference")
    @patch("video_generation.pipeline.load_env")
    def test_analyze_step(
        self,
        _mock_load_env: MagicMock,
        mock_analyze: MagicMock,
        product_dir: Path,
        tiny_video: Path,
        storage_root: Path,
    ) -> None:
        cfg = _config(product_dir, storage_root, reference_video=tiny_video)
        mock_analyze.return_value = AnalysisResult(
            analysis_text="done",
            output_path=storage_root / "a.md",
            content_id="aid",
        )
        run_step("analyze", cfg)
        mock_analyze.assert_called_once()

    @patch("video_generation.pipeline.load_env")
    def test_analyze_step_requires_video(
        self,
        _mock_load_env: MagicMock,
        product_dir: Path,
        analysis_file: Path,
        storage_root: Path,
    ) -> None:
        cfg = _config(product_dir, storage_root, reference_analysis=analysis_file)
        with pytest.raises(ValueError, match="--reference-video is required"):
            run_step("analyze", cfg)

    @patch("video_generation.pipeline.load_env")
    def test_unknown_step_raises(
        self,
        _mock_load_env: MagicMock,
        product_dir: Path,
        analysis_file: Path,
        storage_root: Path,
    ) -> None:
        cfg = _config(product_dir, storage_root, reference_analysis=analysis_file)
        with pytest.raises(ValueError, match="Unknown step"):
            run_step("nonexistent", cfg)

    @patch("video_generation.pipeline.write_script")
    @patch("video_generation.pipeline.load_env")
    def test_script_step(
        self,
        _mock_load_env: MagicMock,
        mock_write: MagicMock,
        product_dir: Path,
        analysis_file: Path,
        storage_root: Path,
    ) -> None:
        cfg = _config(product_dir, storage_root, reference_analysis=analysis_file)
        mock_write.return_value = ScriptResult(
            script_text="s",
            script_path=storage_root / "script.md",
            content_id="sid",
        )
        run_step("script", cfg)
        mock_write.assert_called_once()
