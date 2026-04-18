"""Step 1: Analyze a reference video using Gemini native video understanding."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from google.genai import types

from video_generation.clients import get_gemini_client
from video_generation.config import AnalysisResult
from video_generation.prompts import load_prompt, load_system_prompt
from video_generation.store import StepContext

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
FILE_POLL_INTERVAL_SECONDS = 5
STEP_NAME = "analyze"
OUTPUT_NAME = "analysis"


def analyze_reference(
    reference_video_content_id: str,
    ctx: StepContext,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    num_frames: int = 20,  # noqa: ARG001  - kept for backwards compat with the CLI
) -> AnalysisResult:
    """Analyze a reference video and produce a shot breakdown.

    Materializes the reference video bytes from the content store to a
    temp file, uploads to the Gemini File API, asks Gemini for a professional
    cinematography analysis, and registers the result as a step output.

    Args:
        reference_video_content_id: Content id of the reference video.
        ctx: Step context bound to the active run + step.
        gemini_model: Gemini model identifier.
        num_frames: Unused, kept for backward compatibility with the CLI.

    Returns:
        AnalysisResult with analysis text, content_id, and (when local) a
        path to the convenience copy under runs/<run_id>/steps/analyze/.
    """
    ctx.begin()

    client = get_gemini_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = ctx.materialize_input(reference_video_content_id, Path(tmpdir))
        logger.info("Analyzing reference video (content=%s)", reference_video_content_id[:16])

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
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )

        analysis = response.text or ""
        if not analysis:
            raise RuntimeError("Gemini returned an empty analysis")
        logger.info("Analysis complete")

        try:
            client.files.delete(name=file_name)
            logger.debug("Cleaned up uploaded file: %s", file_name)
        except (OSError, RuntimeError):
            logger.debug("Could not delete uploaded file (non-critical)")

    content_id = ctx.record(
        name=OUTPUT_NAME,
        data=analysis.encode("utf-8"),
        original_name="analysis.md",
        mime="text/markdown",
        kind="text",
    )
    ctx.set_attribute("gemini_model", gemini_model)
    ctx.end()

    output_path = ctx.run_store.local_step_output_path(ctx.run_id, STEP_NAME, OUTPUT_NAME)
    return AnalysisResult(
        analysis_text=analysis,
        output_path=output_path or Path(""),
        content_id=content_id,
    )
