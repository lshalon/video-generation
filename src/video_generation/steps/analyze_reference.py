"""Step 1: Analyze a reference video using Claude vision."""

import base64
import logging
from pathlib import Path

import cv2
import numpy as np

from video_generation.clients import get_anthropic_client
from video_generation.config import AnalysisResult
from video_generation.prompts import load_prompt, load_system_prompt

logger = logging.getLogger(__name__)


def extract_frames(video_path: Path, num_frames: int = 20) -> list[np.ndarray]:
    """Extract evenly-spaced frames from a video.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract.

    Returns:
        List of frames as numpy arrays (BGR format).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    logger.info("Video info: %d frames, %.1f FPS, %.1fs duration", total_frames, fps, duration)

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            logger.warning("Could not read frame %d", idx)

    cap.release()
    logger.info("Extracted %d frames", len(frames))
    return frames


def frame_to_base64(frame: np.ndarray, max_size: int = 1024) -> str:
    """Convert a frame to base64-encoded JPEG.

    Args:
        frame: Frame as numpy array (BGR format).
        max_size: Maximum dimension (width or height).

    Returns:
        Base64-encoded JPEG string.
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buffer).decode("utf-8")


def analyze_reference(
    video_path: Path,
    *,
    claude_model: str = "claude-sonnet-4-20250514",
    num_frames: int = 20,
    output_dir: Path | None = None,
) -> AnalysisResult:
    """Analyze a reference video and produce a shot breakdown.

    Extracts frames from the video, sends them to Claude with the
    video_analyst system prompt, and saves the resulting analysis.

    Args:
        video_path: Path to the reference video (.mp4).
        claude_model: Claude model identifier.
        num_frames: Number of frames to extract.
        output_dir: Directory for the analysis output. Defaults to
            the same directory as the video.

    Returns:
        AnalysisResult with analysis text and output path.
    """
    logger.info("Analyzing reference video: %s", video_path)

    frames = extract_frames(video_path, num_frames)

    frame_images = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_to_base64(frame),
            },
        }
        for frame in frames
    ]

    system_prompt = load_system_prompt("video_analyst")
    user_prompt = load_prompt("shot_breakdown_request", category="examples")

    content = frame_images + [
        {
            "type": "text",
            "text": (
                f"These are {len(frames)} frames extracted from a video, "
                f"shown in chronological order.\n\n{user_prompt}"
            ),
        }
    ]

    client = get_anthropic_client()
    logger.info("Sending %d frames to Claude for analysis...", len(frames))

    response = client.messages.create(
        model=claude_model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],  # type: ignore[typeddict-item]
    )

    analysis: str = response.content[0].text  # type: ignore[union-attr]
    logger.info(
        "Analysis complete (input: %d, output: %d tokens)",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    if output_dir is None:
        output_dir = video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    output_path = output_dir / f"{stem}-analysis.md"
    output_path.write_text(analysis)
    logger.info("Analysis saved to: %s", output_path)

    return AnalysisResult(analysis_text=analysis, output_path=output_path)
