"""CLI entry point: python -m video_generation"""

import argparse
import logging
from pathlib import Path

from video_generation.config import DEFAULT_STORAGE_SPEC, PipelineConfig
from video_generation.pipeline import run_pipeline, run_step

VALID_STEPS = ("analyze", "script", "frame", "video")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m video_generation",
        description="End-to-end video generation pipeline.",
    )
    parser.add_argument(
        "--product-dir",
        type=Path,
        required=True,
        help="Directory containing product images (.webp, .jpg, .png). Files are auto-registered into the content store.",
    )
    parser.add_argument(
        "--reference-video",
        type=Path,
        default=None,
        help="Path to a reference video (.mp4) to analyze.",
    )
    parser.add_argument(
        "--reference-analysis",
        type=Path,
        default=None,
        help="Path to an existing reference analysis (.md). Skips the analyze step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs"),
        help="(Legacy) Output directory. Outputs are now organized under <storage>/runs/<run_id>/.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=DEFAULT_STORAGE_SPEC,
        help=(
            "Storage spec, e.g. 'local:./data' (default) or 'gdrive:<folder-id>'. "
            "Determines where the content library and run records are persisted."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Inspect/resume a specific run id. When set, the run id is used to "
            "resolve the manifest; pipeline execution still skips completed steps."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="",
        help="Free-form fork tag added to the run params; same inputs/code with a different variant fork into a separate run id.",
    )
    parser.add_argument(
        "--edit-model",
        type=str,
        default="fal-ai/nano-banana-2/edit",
        help="Fal endpoint for image editing/compositing (default: Nano Banana 2 Edit).",
    )
    parser.add_argument(
        "--video-model",
        type=str,
        default="fal-ai/kling-video/v2.6/pro/image-to-video",
        help="Fal endpoint for video generation.",
    )
    parser.add_argument(
        "--video-duration",
        type=str,
        default="5",
        help='Video duration in seconds (default: "5").',
    )
    parser.add_argument(
        "--claude-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model identifier (used by the video motion-prompt step).",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model identifier (used for analysis, script, scene prompt, and critique).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of frames to extract from reference video (default: 20). Unused with Gemini.",
    )
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=3,
        help="Max critique-and-correct iterations during starting-frame generation (default: 3).",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=VALID_STEPS,
        default=None,
        help="Run a single step instead of the full pipeline.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from parsed CLI arguments."""
    return PipelineConfig(
        product_dir=args.product_dir,
        reference_video=args.reference_video,
        reference_analysis=args.reference_analysis,
        output_dir=args.output_dir,
        storage_spec=args.storage,
        run_id=args.run_id,
        variant=args.variant,
        claude_model=args.claude_model,
        gemini_model=args.gemini_model,
        edit_model=args.edit_model,
        video_model=args.video_model,
        video_duration=args.video_duration,
        num_frames=args.num_frames,
        max_refinements=args.max_refinements,
    )


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    config = build_config(args)

    if args.step:
        run_step(args.step, config)
    else:
        result = run_pipeline(config)
        print(result.summary())


if __name__ == "__main__":
    main()
