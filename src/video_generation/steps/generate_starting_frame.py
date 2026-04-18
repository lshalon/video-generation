"""Step 3: Generate a starting frame via multi-stage image generation.

Stage 1: Claude writes a scene prompt, Seedream v4 generates the base scene
         WITHOUT the product (text-to-image).
Stage 2: Gemini sees both the scene and product photos and writes a composite
         prompt. The edit model composites the product onto the scene.
Stage 3: Iterative refinement — Gemini critiques the composite against the
         product references and the edit model corrects until acceptable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template
from typing import Any

import fal_client
import httpx
from google.genai import types
from pydantic import BaseModel

from video_generation.clients import get_anthropic_client, get_gemini_client
from video_generation.config import StartingFrameResult
from video_generation.media import save_image
from video_generation.prompts import load_prompt
from video_generation.steps.write_script import load_product_images

logger = logging.getLogger(__name__)

SEEDREAM_TEXT_TO_IMAGE = "fal-ai/bytedance/seedream/v4/text-to-image"
DEFAULT_EDIT_MODEL = "fal-ai/bytedance/seedream/v4.5/edit"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"

SCENE_PROMPT_SYSTEM = (
    "You are an expert at writing prompts for AI image generation models. "
    "Write clear, detailed prompts that produce high-quality results. "
    "The scene should NOT include any jewelry - it will be added later.\n\n"
    "IMPORTANT: Avoid words that might trigger content filters like "
    '"naked", "bare", "exposed". Instead use words like "unadorned", '
    '"without jewelry", "empty earlobes".'
)

MAX_REFINEMENTS = 3


def generate_starting_frame(
    script: str,
    product_dir: Path,
    *,
    claude_model: str = "claude-sonnet-4-20250514",
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    edit_model: str = DEFAULT_EDIT_MODEL,
    max_refinements: int = MAX_REFINEMENTS,
    output_dir: Path | None = None,
) -> StartingFrameResult:
    """Generate a starting frame image for video generation.

    Stages:
    1. Claude writes a scene prompt, Seedream generates the base scene.
    2. Gemini writes a composite prompt, edit model composites the product.
    3. Iterative refinement: Gemini critiques, edit model corrects (up to
       max_refinements times).

    Args:
        script: The video script text.
        product_dir: Directory containing product images.
        claude_model: Claude model for scene prompt generation (Stage 1).
        gemini_model: Gemini model for composite prompt + critique (Stages 2–3).
        edit_model: Fal endpoint for image editing/compositing.
        max_refinements: Maximum number of critique-and-correct iterations.
        output_dir: Directory for output images.

    Returns:
        StartingFrameResult with the path to the saved frame.
    """
    logger.info("Generating starting frame (with up to %d refinement rounds)", max_refinements)

    anthropic_client = get_anthropic_client()
    save_dir = output_dir or Path("data/outputs/images")
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Generate base scene (Claude + Seedream) ---
    scene_bytes = _generate_base_scene(script, claude_model, anthropic_client)

    # --- Stage 2: Initial composite (Gemini + Seedream Edit) ---
    product_images = load_product_images(product_dir)
    composite_prompt = _write_composite_prompt(scene_bytes, product_images, gemini_model)
    logger.info("Composite prompt: %s", composite_prompt[:200])
    (save_dir / "composite_prompt.txt").write_text(composite_prompt)

    product_urls = _upload_product_images(product_images)
    scene_url = fal_client.upload(scene_bytes, content_type="image/png")

    composite_bytes = _run_edit(edit_model, composite_prompt, [scene_url, *product_urls])

    saved_path = save_image(
        composite_bytes, prefix="starting_frame_v0", extension="png", output_dir=save_dir
    )
    logger.info("Initial composite saved: %s", saved_path)

    # --- Stage 3: Iterative refinement (Gemini critique loop) ---
    current_bytes = composite_bytes
    for i in range(1, max_refinements + 1):
        critique = _critique_composite(current_bytes, product_images, gemini_model)
        logger.info("Refinement %d critique: %s", i, critique)

        critique_path = save_dir / f"critique_v{i}.json"
        critique_path.write_text(json.dumps(critique, indent=2))

        if critique.get("acceptable", False):
            logger.info("Composite accepted after %d refinement(s)", i - 1)
            break

        correction = critique.get("correction_prompt", "")
        if not correction:
            logger.info("No correction prompt provided, stopping refinement")
            break

        logger.info("Refinement %d: %s", i, correction)

        current_url = fal_client.upload(current_bytes, content_type="image/png")
        current_bytes = _run_edit(edit_model, correction, [current_url])

        saved_path = save_image(
            current_bytes, prefix=f"starting_frame_v{i}", extension="png", output_dir=save_dir
        )
        logger.info("Refinement %d saved: %s", i, saved_path)
    else:
        logger.warning("Reached max refinements (%d) without acceptance", max_refinements)

    # Save the final version with the standard prefix
    final_path = save_image(
        current_bytes, prefix="starting_frame", extension="png", output_dir=output_dir
    )
    logger.info("Final starting frame saved: %s", final_path)

    return StartingFrameResult(frame_path=final_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_base_scene(script: str, claude_model: str, client: Any) -> bytes:
    """Have Claude write a scene prompt, then generate with Seedream."""
    scene_template = load_prompt("scene_without_product", category="examples")
    scene_request = Template(scene_template).substitute(script=script)

    logger.info("Generating scene prompt with Claude (no product)...")
    scene_prompt_response = client.messages.create(
        model=claude_model,
        max_tokens=1024,
        system=SCENE_PROMPT_SYSTEM,
        messages=[{"role": "user", "content": scene_request}],
    )
    scene_generation_prompt: str = scene_prompt_response.content[0].text  # type: ignore[union-attr]
    logger.info("Scene prompt: %s", scene_generation_prompt[:120])

    logger.info("Generating base scene with Seedream...")
    scene_result = fal_client.subscribe(
        SEEDREAM_TEXT_TO_IMAGE,
        arguments={
            "prompt": scene_generation_prompt,
            "image_size": "portrait_4_3",
            "num_images": 1,
        },
        with_logs=True,
    )
    scene_image_url = scene_result["images"][0]["url"]
    logger.info("Base scene URL: %s", scene_image_url)

    scene_response = httpx.get(scene_image_url)
    scene_response.raise_for_status()
    return bytes(scene_response.content)


def _write_composite_prompt(
    scene_bytes: bytes,
    product_images: list[Path],
    gemini_model: str,
) -> str:
    """Send scene + all product images to Gemini and get a composite prompt."""
    prompt_text = load_prompt("write_composite_prompt", category="examples")

    content: list[types.Part | str] = [
        types.Part.from_bytes(data=scene_bytes, mime_type="image/png"),
    ]

    for path in product_images:
        img_bytes = path.read_bytes()
        suffix = path.suffix.lstrip(".").lower()
        mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
        content.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

    content.append(prompt_text)

    client = get_gemini_client()
    logger.info(
        "Writing composite prompt with Gemini %s (scene + %d product images)...",
        gemini_model,
        len(product_images),
    )
    response = client.models.generate_content(
        model=gemini_model,
        contents=content,
        config=types.GenerateContentConfig(max_output_tokens=1024),
    )

    result = response.text or ""
    if not result:
        raise RuntimeError("Gemini returned an empty composite prompt")
    return result


class CompositesCritique(BaseModel):
    """Structured critique of a composite image."""

    acceptable: bool
    issues: list[str]
    correction_prompt: str


def _critique_composite(
    composite_bytes: bytes,
    product_images: list[Path],
    gemini_model: str,
) -> dict:
    """Send the composite + product references to Gemini for structured critique."""
    prompt_text = load_prompt("critique_composite", category="examples")

    content: list[types.Part | str] = [
        types.Part.from_bytes(data=composite_bytes, mime_type="image/png"),
    ]

    for path in product_images:
        img_bytes = path.read_bytes()
        suffix = path.suffix.lstrip(".").lower()
        mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
        content.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

    content.append(prompt_text)

    client = get_gemini_client()
    logger.info("Critiquing composite with Gemini %s (structured output)...", gemini_model)
    response = client.models.generate_content(
        model=gemini_model,
        contents=content,
        config=types.GenerateContentConfig(
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema=CompositesCritique,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
    )

    raw = response.text or ""
    if not raw:
        raise RuntimeError("Gemini returned an empty critique response")
    critique = CompositesCritique.model_validate_json(raw)
    return dict(critique.model_dump())


def _run_edit(edit_model: str, prompt: str, image_urls: list[str]) -> bytes:
    """Run the edit model and return the result image bytes."""
    logger.info("Running edit model %s...", edit_model)
    result = fal_client.subscribe(
        edit_model,
        arguments={
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": "portrait_4_3",
            "num_images": 1,
        },
        with_logs=True,
    )
    image_url = result["images"][0]["url"]
    response = httpx.get(image_url)
    response.raise_for_status()
    return bytes(response.content)


def _upload_product_images(product_images: list[Path]) -> list[str]:
    """Upload all product images to fal. Images with '_on' sorted first."""
    sorted_images = sorted(
        product_images,
        key=lambda p: (0 if "_on" in p.stem.lower() else 1, p.name),
    )
    urls = []
    for img_path in sorted_images:
        suffix = img_path.suffix.lstrip(".")
        ct = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
        urls.append(fal_client.upload(img_path.read_bytes(), content_type=ct))
        logger.info("  Uploaded product image: %s", img_path.name)
    return urls
