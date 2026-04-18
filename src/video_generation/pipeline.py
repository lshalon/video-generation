"""Pipeline orchestrator for end-to-end video generation."""

import logging
from pathlib import Path

from video_generation.clients import load_env
from video_generation.config import (
    PipelineConfig,
    PipelineResult,
)
from video_generation.steps.analyze_reference import analyze_reference
from video_generation.steps.generate_starting_frame import generate_starting_frame
from video_generation.steps.generate_video import generate_video
from video_generation.steps.write_script import write_script

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full video generation pipeline.

    Steps:
        1. Analyze reference video (or use existing analysis)
        2. Write a video script
        3. Generate a starting frame image
        4. Generate the final video

    Args:
        config: Pipeline configuration.

    Returns:
        PipelineResult with paths to every intermediate and final artifact.
    """
    load_env()
    config.validate()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Analyze reference (Gemini — native video understanding)
    analysis_path = _resolve_analysis(config, output_dir)

    # Step 2: Write script
    logger.info("=== Step 2: Write Script ===")
    script_result = write_script(
        analysis_path,
        config.product_dir,
        claude_model=config.claude_model,
        output_dir=output_dir,
    )

    # Step 3: Generate starting frame (Gemini for composite prompt + critique)
    logger.info("=== Step 3: Generate Starting Frame ===")
    frame_result = generate_starting_frame(
        script_result.script_text,
        config.product_dir,
        claude_model=config.claude_model,
        gemini_model=config.gemini_model,
        edit_model=config.edit_model,
        output_dir=output_dir / "images",
    )

    # Step 4: Generate video
    logger.info("=== Step 4: Generate Video ===")
    video_result = generate_video(
        frame_result.frame_path,
        script_result.script_text,
        claude_model=config.claude_model,
        video_model=config.video_model,
        duration=config.video_duration,
        output_dir=output_dir / "videos",
    )

    result = PipelineResult(
        analysis_path=analysis_path,
        script_path=script_result.script_path,
        starting_frame_path=frame_result.frame_path,
        video_path=video_result.video_path,
    )
    logger.info(result.summary())
    return result


def run_step(step_name: str, config: PipelineConfig) -> None:
    """Run a single pipeline step.

    Args:
        step_name: One of "analyze", "script", "frame", "video".
        config: Pipeline configuration.
    """
    load_env()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if step_name == "analyze":
        if not config.reference_video:
            raise ValueError("--reference-video is required for the 'analyze' step")
        analyze_reference(
            config.reference_video,
            gemini_model=config.gemini_model,
            num_frames=config.num_frames,
            output_dir=output_dir,
        )

    elif step_name == "script":
        analysis_path = _resolve_analysis(config, output_dir)
        write_script(
            analysis_path,
            config.product_dir,
            claude_model=config.claude_model,
            output_dir=output_dir,
        )

    elif step_name == "frame":
        script_path = output_dir / "script.md"
        if not script_path.exists():
            raise FileNotFoundError(
                f"Script not found at {script_path}. Run the 'script' step first."
            )
        script_text = script_path.read_text()
        generate_starting_frame(
            script_text,
            config.product_dir,
            claude_model=config.claude_model,
            gemini_model=config.gemini_model,
            edit_model=config.edit_model,
            output_dir=output_dir / "images",
        )

    elif step_name == "video":
        script_path = output_dir / "script.md"
        if not script_path.exists():
            raise FileNotFoundError(
                f"Script not found at {script_path}. Run the 'script' step first."
            )
        images_dir = output_dir / "images"
        starting_frames = sorted(images_dir.glob("starting_frame*.png"))
        if not starting_frames:
            raise FileNotFoundError(
                f"No starting frame found in {images_dir}. Run the 'frame' step first."
            )
        frame_path = starting_frames[-1]  # most recent
        generate_video(
            frame_path,
            script_path.read_text(),
            claude_model=config.claude_model,
            video_model=config.video_model,
            duration=config.video_duration,
            output_dir=output_dir / "videos",
        )

    else:
        raise ValueError(
            f"Unknown step: {step_name!r}. Must be one of: analyze, script, frame, video"
        )


def _resolve_analysis(config: PipelineConfig, output_dir: Path) -> Path:
    """Resolve the reference analysis path, running analysis if needed."""
    if config.reference_analysis:
        if not config.reference_analysis.exists():
            raise FileNotFoundError(f"Reference analysis not found: {config.reference_analysis}")
        logger.info("=== Step 1: Using existing analysis: %s ===", config.reference_analysis)
        return config.reference_analysis

    if config.reference_video:
        logger.info("=== Step 1: Analyze Reference Video ===")
        result = analyze_reference(
            config.reference_video,
            gemini_model=config.gemini_model,
            num_frames=config.num_frames,
            output_dir=output_dir,
        )
        return result.output_path

    raise ValueError("Must provide either --reference-video or --reference-analysis")
