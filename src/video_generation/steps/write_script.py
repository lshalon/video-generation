"""Step 2: Generate a video script from reference analysis and product images."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from string import Template
from typing import Any

import cv2
import numpy as np
from google.genai import types

from video_generation.clients import get_gemini_client
from video_generation.config import ScriptResult
from video_generation.prompts import load_prompt, load_system_prompt
from video_generation.store import StepContext

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png")
STEP_NAME = "script"
OUTPUT_NAME = "script"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"


def discover_product_images(product_dir: Path) -> list[Path]:
    """Find all product images in a directory, sorted by filename.

    Used at the input-registration boundary (CLI / migration script). Steps
    themselves consume content ids rather than directories.
    """
    images = [
        p
        for p in product_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    ]
    images.sort(key=lambda p: p.name)
    if not images:
        raise FileNotFoundError(f"No product images found in {product_dir}")
    return images


load_product_images = discover_product_images


def encode_image_bytes_as_base64(data: bytes, max_size: int = 800) -> tuple[str, str]:
    """Decode image bytes, optionally downscale, and re-encode as base64 JPEG."""
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.standard_b64encode(buffer).decode("utf-8")
    return b64, "image/jpeg"


def load_image_as_base64(path: Path, max_size: int = 800) -> tuple[str, str]:
    """Read a file and return its bytes encoded as base64 JPEG."""
    return encode_image_bytes_as_base64(path.read_bytes(), max_size=max_size)


def _resize_image_jpeg(data: bytes, max_size: int = 800) -> bytes:
    """Decode image bytes, downscale to ``max_size`` on the long edge, return JPEG bytes."""
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed for product image")
    return bytes(buffer.tobytes())


def _human_name_from_meta(original_name: str | None, fallback_id: str) -> str:
    if original_name:
        stem = Path(original_name).stem
        return stem.replace("_", " ").title()
    return f"Image {fallback_id[:8]}"


def write_script(
    reference_analysis_content_id: str,
    product_image_content_ids: list[str],
    ctx: StepContext,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> ScriptResult:
    """Generate a video script that emulates a reference style for a product.

    Uses Gemini 3.1 Pro multimodally: the reference analysis is inlined into a
    text prompt, and each product image is attached as inline JPEG bytes.

    Args:
        reference_analysis_content_id: Content id of the reference analysis markdown.
        product_image_content_ids: Content ids of product images (in order).
        ctx: Step context bound to the active run + step.
        gemini_model: Gemini model identifier.

    Returns:
        ScriptResult with script text, content_id, and (when local) the path to
        the convenience copy under runs/<run_id>/steps/script/.
    """
    ctx.begin()

    if not product_image_content_ids:
        raise ValueError("write_script requires at least one product image content id")

    logger.info(
        "Writing script (analysis=%s, %d product images)",
        reference_analysis_content_id[:16],
        len(product_image_content_ids),
    )

    reference_analysis = ctx.content_store.get_bytes(reference_analysis_content_id).decode("utf-8")

    product_description = "Product: Earrings\n\nAvailable product shots:\n"
    image_parts: list[types.Part] = []
    for content_id in product_image_content_ids:
        meta = ctx.content_store.get_meta(content_id)
        product_description += f"- {_human_name_from_meta(meta.original_name, content_id)}\n"
        image_bytes = ctx.content_store.get_bytes(content_id)
        jpeg_bytes = _resize_image_jpeg(image_bytes)
        image_parts.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))

    system_prompt = load_system_prompt("script_writer")
    user_prompt_template = load_prompt("emulate_reference_script", category="examples")
    user_prompt = Template(user_prompt_template).substitute(
        reference_analysis=reference_analysis,
        product_description=product_description,
    )

    contents: list[Any] = list(image_parts)
    contents.append(f"Here are the product images I have available.\n\n{user_prompt}")

    client = get_gemini_client()
    logger.info("Generating script with Gemini %s...", gemini_model)

    response = client.models.generate_content(
        model=gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
    )

    script: str = response.text or ""
    if not script:
        raise RuntimeError("Gemini returned an empty script")

    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    logger.info(
        "Script complete (input: %d, output: %d tokens)",
        input_tokens,
        output_tokens,
    )

    content_id = ctx.record(
        name=OUTPUT_NAME,
        data=script.encode("utf-8"),
        original_name="script.md",
        mime="text/markdown",
        kind="text",
    )
    ctx.set_attribute("gemini_model", gemini_model)
    ctx.set_attribute("input_tokens", input_tokens)
    ctx.set_attribute("output_tokens", output_tokens)
    ctx.end()

    output_path = ctx.run_store.local_step_output_path(ctx.run_id, STEP_NAME, OUTPUT_NAME)
    return ScriptResult(
        script_text=script,
        script_path=output_path or Path(""),
        content_id=content_id,
    )
