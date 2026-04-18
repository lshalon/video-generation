"""Tests for config dataclasses and CLI argument parsing."""

from pathlib import Path

import pytest

from video_generation.config import (
    DEFAULT_STORAGE_SPEC,
    AnalysisResult,
    PipelineConfig,
    PipelineResult,
    ScriptResult,
    StartingFrameResult,
    VideoResult,
)


class TestPipelineConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        product_dir = tmp_path / "product"
        product_dir.mkdir()
        ref = tmp_path / "analysis.md"
        ref.write_text("test")

        cfg = PipelineConfig(
            product_dir=product_dir,
            reference_analysis=ref,
        )
        assert cfg.output_dir == Path("data/outputs")
        assert cfg.video_duration == "5"
        assert cfg.num_frames == 20
        assert cfg.claude_model == "claude-sonnet-4-20250514"
        assert cfg.gemini_model == "gemini-3.1-pro-preview"
        assert cfg.storage_spec == DEFAULT_STORAGE_SPEC
        assert cfg.run_id is None
        assert cfg.variant == ""

    def test_validate_missing_both_references(self, tmp_path: Path) -> None:
        product_dir = tmp_path / "product"
        product_dir.mkdir()

        cfg = PipelineConfig(product_dir=product_dir)
        with pytest.raises(ValueError, match="--reference-video or --reference-analysis"):
            cfg.validate()

    def test_validate_missing_product_dir(self, tmp_path: Path) -> None:
        cfg = PipelineConfig(
            product_dir=tmp_path / "nonexistent",
            reference_analysis=tmp_path / "a.md",
        )
        with pytest.raises(FileNotFoundError, match="Product directory"):
            cfg.validate()

    def test_validate_missing_reference_video(self, tmp_path: Path) -> None:
        product_dir = tmp_path / "product"
        product_dir.mkdir()

        cfg = PipelineConfig(
            product_dir=product_dir,
            reference_video=tmp_path / "missing.mp4",
        )
        with pytest.raises(FileNotFoundError, match="Reference video"):
            cfg.validate()

    def test_validate_missing_reference_analysis(self, tmp_path: Path) -> None:
        product_dir = tmp_path / "product"
        product_dir.mkdir()

        cfg = PipelineConfig(
            product_dir=product_dir,
            reference_analysis=tmp_path / "missing.md",
        )
        with pytest.raises(FileNotFoundError, match="Reference analysis"):
            cfg.validate()

    def test_validate_passes(self, tmp_path: Path) -> None:
        product_dir = tmp_path / "product"
        product_dir.mkdir()
        ref = tmp_path / "analysis.md"
        ref.write_text("test")

        cfg = PipelineConfig(
            product_dir=product_dir,
            reference_analysis=ref,
        )
        cfg.validate()


class TestResultDataclasses:
    def test_analysis_result(self, tmp_path: Path) -> None:
        r = AnalysisResult(analysis_text="hello", output_path=tmp_path / "a.md", content_id="cid")
        assert r.analysis_text == "hello"
        assert r.content_id == "cid"

    def test_script_result(self, tmp_path: Path) -> None:
        r = ScriptResult(script_text="s", script_path=tmp_path / "s.md", content_id="sid")
        assert r.script_text == "s"
        assert r.content_id == "sid"

    def test_starting_frame_result(self, tmp_path: Path) -> None:
        r = StartingFrameResult(frame_path=tmp_path / "frame.png", content_id="fid")
        assert r.frame_path.name == "frame.png"
        assert r.content_id == "fid"

    def test_video_result(self, tmp_path: Path) -> None:
        r = VideoResult(video_path=tmp_path / "v.mp4", content_id="vid")
        assert r.video_path.name == "v.mp4"
        assert r.content_id == "vid"

    def test_pipeline_result_summary_includes_run(self, tmp_path: Path) -> None:
        r = PipelineResult(
            analysis_path=tmp_path / "a.md",
            script_path=tmp_path / "s.md",
            starting_frame_path=tmp_path / "f.png",
            video_path=tmp_path / "v.mp4",
            run_id="abc123",
        )
        assert "abc123" in r.summary()


class TestCLIParsing:
    def test_required_args(self) -> None:
        from video_generation.__main__ import parse_args

        args = parse_args(
            [
                "--product-dir",
                "data/inputs/jewelry/product",
                "--reference-analysis",
                "data/inputs/ref.md",
            ]
        )
        assert args.product_dir == Path("data/inputs/jewelry/product")
        assert args.reference_analysis == Path("data/inputs/ref.md")
        assert args.step is None
        assert args.verbose is False
        assert args.storage == DEFAULT_STORAGE_SPEC
        assert args.run_id is None

    def test_step_arg(self) -> None:
        from video_generation.__main__ import parse_args

        args = parse_args(
            [
                "--product-dir",
                "p",
                "--step",
                "analyze",
                "--reference-video",
                "v.mp4",
            ]
        )
        assert args.step == "analyze"

    def test_storage_and_run_id_flags(self) -> None:
        from video_generation.__main__ import parse_args

        args = parse_args(
            [
                "--product-dir",
                "p",
                "--reference-analysis",
                "a.md",
                "--storage",
                "gdrive:my-folder",
                "--run-id",
                "abc123",
                "--variant",
                "alt",
            ]
        )
        assert args.storage == "gdrive:my-folder"
        assert args.run_id == "abc123"
        assert args.variant == "alt"

    def test_verbose_flag(self) -> None:
        from video_generation.__main__ import parse_args

        args = parse_args(["--product-dir", "p", "--reference-analysis", "a.md", "-v"])
        assert args.verbose is True

    def test_build_config(self) -> None:
        from video_generation.__main__ import build_config, parse_args

        args = parse_args(
            [
                "--product-dir",
                "/tmp/p",
                "--reference-analysis",
                "/tmp/a.md",
                "--video-model",
                "fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
                "--video-duration",
                "10",
                "--storage",
                "local:./data2",
            ]
        )
        cfg = build_config(args)
        assert cfg.product_dir == Path("/tmp/p")
        assert cfg.video_duration == "10"
        assert "seedance" in cfg.video_model
        assert cfg.storage_spec == "local:./data2"
