"""CLI entry point: python -m video_generation.pipeline"""

import argparse
import logging
from pathlib import Path

from video_generation.config import PipelineConfig
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
        help="Directory containing product images (.webp, .jpg, .png).",
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
        help="Output directory (default: data/outputs).",
    )
    parser.add_argument(
        "--edit-model",
        type=str,
        default="fal-ai/bytedance/seedream/v4.5/edit",
        help="Fal endpoint for image editing/compositing (default: Seedream v4.5).",
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
        help="Claude model identifier.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model identifier (used for video analysis and image critique).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of frames to extract from reference video (default: 20). Unused with Gemini.",
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
        claude_model=args.claude_model,
        gemini_model=args.gemini_model,
        edit_model=args.edit_model,
        video_model=args.video_model,
        video_duration=args.video_duration,
        num_frames=args.num_frames,
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
