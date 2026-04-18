"""Step 4: Generate a video from a starting frame and script."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from string import Template

import cv2
import fal_client
import httpx
import numpy as np

from video_generation.clients import get_anthropic_client
from video_generation.config import VideoResult
from video_generation.prompts import load_prompt
from video_generation.store import StepContext

logger = logging.getLogger(__name__)

SEEDANCE_PREFIX = "fal-ai/bytedance/seedance"
STEP_NAME = "video"
OUTPUT_NAME = "video"


def generate_video(
    starting_frame_content_id: str,
    script_content_id: str,
    ctx: StepContext,
    claude_model: str = "claude-sonnet-4-20250514",
    video_model: str = "fal-ai/kling-video/v2.6/pro/image-to-video",
    duration: str = "5",
) -> VideoResult:
    """Generate a video from a starting frame image and script.

    Uses Claude to create a motion prompt, then calls the video model
    (Kling or Seedance) via fal to produce the video. The motion prompt and
    the final mp4 are both registered as outputs of this step.

    Args:
        starting_frame_content_id: Content id of the starting frame image.
        script_content_id: Content id of the video script.
        ctx: Step context bound to the active run + step.
        claude_model: Claude model identifier.
        video_model: Fal endpoint for video generation.
        duration: Video duration in seconds (as string).

    Returns:
        VideoResult with content_id and (when local) a path to the
        convenience copy.
    """
    ctx.begin()

    starting_frame_bytes = ctx.content_store.get_bytes(starting_frame_content_id)
    script = ctx.content_store.get_bytes(script_content_id).decode("utf-8")
    logger.info(
        "Generating video (frame=%s, script=%s)",
        starting_frame_content_id[:16],
        script_content_id[:16],
    )

    motion_template = load_prompt("video_motion_prompt", category="examples")
    motion_request = Template(motion_template).substitute(
        script=script,
        starting_frame_description=(
            "Model in portrait orientation with visible ear, luxury jewelry commercial aesthetic"
        ),
    )

    frame_b64 = _compress_frame_for_claude(starting_frame_bytes)
    logger.info("Generating motion prompt with Claude...")

    client = get_anthropic_client()
    motion_response = client.messages.create(
        model=claude_model,
        max_tokens=256,
        system=(
            "You write concise motion prompts for AI video generation. "
            "Keep prompts short (1-2 sentences) and focus only on movement."
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Here is the starting frame.\n\n{motion_request}",
                    },
                ],
            }
        ],
    )

    video_prompt: str = motion_response.content[0].text  # type: ignore[union-attr]
    logger.info("Motion prompt: %s", video_prompt)

    ctx.record(
        name="motion_prompt",
        data=video_prompt.encode("utf-8"),
        original_name="motion_prompt.txt",
        mime="text/plain",
        kind="text",
    )

    upload_bytes = _compress_frame_for_upload(starting_frame_bytes)
    starting_frame_url = fal_client.upload(upload_bytes, content_type="image/jpeg")
    logger.info("Uploaded starting frame to fal")

    is_seedance = video_model.startswith(SEEDANCE_PREFIX)
    arguments = _build_video_arguments(
        video_prompt, starting_frame_url, duration, is_seedance=is_seedance
    )

    logger.info("Generating %ss video with %s...", duration, video_model)
    video_result = fal_client.subscribe(
        video_model,
        arguments=arguments,
        with_logs=True,
    )

    video_url = video_result["video"]["url"]
    logger.info("Video URL: %s", video_url)

    video_response = httpx.get(video_url)
    video_response.raise_for_status()
    video_bytes = bytes(video_response.content)

    content_id = ctx.record(
        name=OUTPUT_NAME,
        data=video_bytes,
        original_name="video.mp4",
        mime="video/mp4",
        kind="video",
    )
    ctx.set_attribute("video_model", video_model)
    ctx.set_attribute("duration_seconds", duration)
    ctx.set_attribute("claude_model", claude_model)
    ctx.end()

    output_path = ctx.run_store.local_step_output_path(ctx.run_id, STEP_NAME, OUTPUT_NAME)
    return VideoResult(
        video_path=output_path or Path(""),
        content_id=content_id,
    )


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode image bytes to a numpy array, raising on failure."""
    img_array = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def _compress_frame_for_claude(raw_bytes: bytes, max_dim: int = 1024) -> str:
    """Compress a frame image to base64 JPEG for Claude (max ~5MB)."""
    img = _decode_image(raw_bytes)

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buffer).decode("utf-8")


def _compress_frame_for_upload(raw_bytes: bytes, max_dim: int = 1920) -> bytes:
    """Compress a frame image to JPEG bytes for fal upload (max ~10MB)."""
    img = _decode_image(raw_bytes)

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return bytes(buffer.tobytes())


def _build_video_arguments(
    prompt: str,
    image_url: str,
    duration: str,
    is_seedance: bool = False,
) -> dict:
    """Build the arguments dict for the video model API call.

    Kling uses ``start_image_url``; Seedance uses ``image_url``.
    """
    if is_seedance:
        return {
            "prompt": prompt,
            "image_url": image_url,
            "duration": duration,
            "generate_audio": False,
        }
    return {
        "prompt": prompt,
        "start_image_url": image_url,
        "duration": duration,
        "negative_prompt": "blur, distort, low quality, jitter, shake",
        "generate_audio": False,
    }
