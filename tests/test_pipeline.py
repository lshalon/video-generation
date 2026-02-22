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


@pytest.fixture
def valid_config(product_dir: Path, analysis_file: Path, tmp_output_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        product_dir=product_dir,
        reference_analysis=analysis_file,
        output_dir=tmp_output_dir,
    )


@pytest.fixture
def video_config(product_dir: Path, tiny_video: Path, tmp_output_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        product_dir=product_dir,
        reference_video=tiny_video,
        output_dir=tmp_output_dir,
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
        valid_config: PipelineConfig,
        tmp_output_dir: Path,
    ) -> None:
        script_path = tmp_output_dir / "script.md"
        frame_path = tmp_output_dir / "images" / "frame.png"
        video_path = tmp_output_dir / "videos" / "video.mp4"

        mock_write_script.return_value = ScriptResult(
            script_text="Test script", script_path=script_path
        )
        mock_gen_frame.return_value = StartingFrameResult(frame_path=frame_path)
        mock_gen_video.return_value = VideoResult(video_path=video_path)

        result = run_pipeline(valid_config)

        assert result.video_path == video_path
        mock_write_script.assert_called_once()
        mock_gen_frame.assert_called_once()
        mock_gen_video.assert_called_once()

        # Verify script output is passed to frame generation
        frame_call_args = mock_gen_frame.call_args
        assert frame_call_args.args[0] == "Test script"

        # Verify frame output is passed to video generation
        video_call_args = mock_gen_video.call_args
        assert video_call_args.args[0] == frame_path

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
        video_config: PipelineConfig,
        tmp_output_dir: Path,
    ) -> None:
        analysis_path = tmp_output_dir / "analysis.md"
        analysis_path.write_text("analysis")

        mock_analyze.return_value = AnalysisResult(
            analysis_text="analysis", output_path=analysis_path
        )
        mock_write_script.return_value = ScriptResult(
            script_text="script", script_path=tmp_output_dir / "script.md"
        )
        mock_gen_frame.return_value = StartingFrameResult(frame_path=tmp_output_dir / "f.png")
        mock_gen_video.return_value = VideoResult(video_path=tmp_output_dir / "v.mp4")

        run_pipeline(video_config)

        mock_analyze.assert_called_once()
        mock_write_script.assert_called_once()
        assert mock_write_script.call_args.args[0] == analysis_path

    @patch("video_generation.pipeline.load_env")
    def test_missing_references_raises(
        self, _mock_load_env: MagicMock, product_dir: Path, tmp_output_dir: Path
    ) -> None:
        cfg = PipelineConfig(product_dir=product_dir, output_dir=tmp_output_dir)
        with pytest.raises(ValueError):
            run_pipeline(cfg)


class TestRunStep:
    @patch("video_generation.pipeline.analyze_reference")
    @patch("video_generation.pipeline.load_env")
    def test_analyze_step(
        self,
        _mock_load_env: MagicMock,
        mock_analyze: MagicMock,
        video_config: PipelineConfig,
        tmp_output_dir: Path,
    ) -> None:
        mock_analyze.return_value = AnalysisResult(
            analysis_text="done", output_path=tmp_output_dir / "a.md"
        )
        run_step("analyze", video_config)
        mock_analyze.assert_called_once()

    @patch("video_generation.pipeline.load_env")
    def test_analyze_step_requires_video(
        self,
        _mock_load_env: MagicMock,
        valid_config: PipelineConfig,
    ) -> None:
        with pytest.raises(ValueError, match="--reference-video is required"):
            run_step("analyze", valid_config)

    @patch("video_generation.pipeline.load_env")
    def test_unknown_step_raises(
        self, _mock_load_env: MagicMock, valid_config: PipelineConfig
    ) -> None:
        with pytest.raises(ValueError, match="Unknown step"):
            run_step("nonexistent", valid_config)

    @patch("video_generation.pipeline.write_script")
    @patch("video_generation.pipeline.load_env")
    def test_script_step(
        self,
        _mock_load_env: MagicMock,
        mock_write: MagicMock,
        valid_config: PipelineConfig,
        tmp_output_dir: Path,
    ) -> None:
        mock_write.return_value = ScriptResult(
            script_text="s", script_path=tmp_output_dir / "script.md"
        )
        run_step("script", valid_config)
        mock_write.assert_called_once()

    @patch("video_generation.pipeline.generate_starting_frame")
    @patch("video_generation.pipeline.load_env")
    def test_frame_step_requires_script(
        self,
        _mock_load_env: MagicMock,
        _mock_gen_frame: MagicMock,
        valid_config: PipelineConfig,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Script not found"):
            run_step("frame", valid_config)

    @patch("video_generation.pipeline.generate_video")
    @patch("video_generation.pipeline.load_env")
    def test_video_step_requires_script(
        self,
        _mock_load_env: MagicMock,
        _mock_gen_video: MagicMock,
        valid_config: PipelineConfig,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Script not found"):
            run_step("video", valid_config)
