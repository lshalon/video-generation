"""Step 1: Analyze a reference video using Gemini native video understanding."""

import logging
import time
from pathlib import Path

from google.genai import types

from video_generation.clients import get_gemini_client
from video_generation.config import AnalysisResult
from video_generation.prompts import load_prompt, load_system_prompt

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
FILE_POLL_INTERVAL_SECONDS = 5


def analyze_reference(
    video_path: Path,
    *,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    num_frames: int = 20,  # noqa: ARG001  — kept for CLI backward compat
    output_dir: Path | None = None,
) -> AnalysisResult:
    """Analyze a reference video and produce a shot breakdown.

    Uploads the video to the Gemini File API for native video understanding,
    then asks Gemini to produce a professional cinematography analysis.

    Args:
        video_path: Path to the reference video (.mp4).
        gemini_model: Gemini model identifier.
        num_frames: Unused, kept for backward compatibility with CLI.
        output_dir: Directory for the analysis output. Defaults to
            the same directory as the video.

    Returns:
        AnalysisResult with analysis text and output path.
    """
    logger.info("Analyzing reference video: %s", video_path)

    client = get_gemini_client()

    # Upload video via Gemini File API for native processing (1 FPS + audio)
    logger.info("Uploading video to Gemini File API...")
    video_file = client.files.upload(file=str(video_path))
    logger.info("Upload started: %s (state: %s)", video_file.name, video_file.state)

    file_name = video_file.name
    if not file_name:
        raise RuntimeError("Gemini File API did not return a file name")

    while video_file.state == "PROCESSING":
        time.sleep(FILE_POLL_INTERVAL_SECONDS)
        video_file = client.files.get(name=file_name)
        logger.debug("File state: %s", video_file.state)

    if video_file.state != "ACTIVE":
        raise RuntimeError(f"Video file processing failed with state: {video_file.state}")
    logger.info("Video file ready: %s", file_name)

    system_prompt = load_system_prompt("video_analyst")
    user_prompt = load_prompt("shot_breakdown_request", category="examples")

    logger.info("Sending video to Gemini for analysis...")
    response = client.models.generate_content(
        model=gemini_model,
        contents=[
            video_file,
            f"This is a reference video for a commercial.\n\n{user_prompt}",
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=4096,
        ),
    )

    analysis = response.text or ""
    if not analysis:
        raise RuntimeError("Gemini returned an empty analysis")
    logger.info("Analysis complete")

    # Clean up the uploaded file
    try:
        client.files.delete(name=file_name)
        logger.debug("Cleaned up uploaded file: %s", file_name)
    except (OSError, RuntimeError):
        logger.debug("Could not delete uploaded file (non-critical)")

    if output_dir is None:
        output_dir = video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    output_path = output_dir / f"{stem}-analysis.md"
    output_path.write_text(analysis)
    logger.info("Analysis saved to: %s", output_path)

    return AnalysisResult(analysis_text=analysis, output_path=output_path)
