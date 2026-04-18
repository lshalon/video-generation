"""Step 3: Generate a starting frame via multi-stage image generation.

Stage 1: Gemini writes a scene prompt (with the reference video and/or its
         analysis as cinematography context), Seedream v4 generates the base
         scene WITHOUT the product (text-to-image).
Stage 2: Gemini sees both the scene and product photos and writes a composite
         prompt. The edit model composites the product onto the scene.
Stage 3: Iterative refinement - Gemini critiques the composite against the
         product references and the edit model corrects until acceptable.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from string import Template
from typing import Any

import fal_client
import httpx
from google.genai import types
from pydantic import BaseModel

from video_generation.clients import get_gemini_client
from video_generation.config import StartingFrameResult
from video_generation.prompts import load_prompt
from video_generation.store import ContentStore, StepContext

logger = logging.getLogger(__name__)

BASE_SCENE_TEXT_TO_IMAGE = "fal-ai/nano-banana-2"
DEFAULT_EDIT_MODEL = "fal-ai/nano-banana-2/edit"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
FILE_POLL_INTERVAL_SECONDS = 5

STEP_NAME = "frame"
OUTPUT_FINAL = "starting_frame"

SCENE_PROMPT_SYSTEM = (
    "You are an expert at writing prompts for AI image generation models. "
    "Write clear, detailed prompts that produce high-quality results. "
    "The scene must NOT include any jewelry - it will be composited in later. "
    "When a reference video and/or its cinematography analysis are provided, "
    "match the reference's shot framing, camera angle, lighting style, and "
    "model pose closely so the generated frame can stand in as the first "
    "frame of a video that emulates the reference.\n\n"
    "IMPORTANT: Avoid words that might trigger content filters like "
    '"naked", "bare", "exposed". Instead use words like "unadorned", '
    '"without jewelry", "empty earlobes".'
)

MAX_REFINEMENTS = 3


def generate_starting_frame(
    script_content_id: str,
    product_image_content_ids: list[str],
    ctx: StepContext,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    edit_model: str = DEFAULT_EDIT_MODEL,
    max_refinements: int = MAX_REFINEMENTS,
    reference_video_content_id: str | None = None,
    reference_analysis_content_id: str | None = None,
) -> StartingFrameResult:
    """Generate a starting frame image for video generation.

    Stages:
    1. Gemini writes a scene prompt grounded in the reference video / analysis
       (when provided), Seedream generates the base scene WITHOUT the product.
    2. Gemini writes a composite prompt, edit model composites the product.
    3. Iterative refinement: Gemini critiques, edit model corrects (up to
       max_refinements times).

    All intermediates (scene prompt, composite prompt, every critique, every
    composite version, the final frame) are registered as content and
    attached to the step record so they can be inspected post-hoc.

    Args:
        script_content_id: Content id of the video script markdown.
        product_image_content_ids: Content ids of product images.
        ctx: Step context bound to the active run + step.
        gemini_model: Gemini model for scene prompt + composite prompt + critique.
        edit_model: Fal endpoint for image editing/compositing.
        max_refinements: Maximum number of critique-and-correct iterations.
        reference_video_content_id: Optional content id of the reference video.
            When set, the bytes are uploaded to the Gemini File API and passed
            to Stage 1 as additional cinematography context.
        reference_analysis_content_id: Optional content id of the reference
            analysis markdown. When set, the text is inlined into the Stage 1
            user prompt as additional context.

    Returns:
        StartingFrameResult with the final content id, intermediate content
        ids, and (when local) a path to the convenience copy.
    """
    ctx.begin()
    logger.info(
        "Generating starting frame for script=%s (max_refinements=%d)",
        script_content_id[:16],
        max_refinements,
    )

    script = ctx.content_store.get_bytes(script_content_id).decode("utf-8")
    reference_analysis = None
    if reference_analysis_content_id is not None:
        reference_analysis = ctx.content_store.get_bytes(reference_analysis_content_id).decode(
            "utf-8"
        )

    # --- Stage 1: Generate base scene (Gemini + Seedream) ---
    scene_prompt, scene_bytes = _generate_base_scene(
        script=script,
        gemini_model=gemini_model,
        content_store=ctx.content_store,
        reference_video_content_id=reference_video_content_id,
        reference_analysis=reference_analysis,
    )
    ctx.record(
        name="scene_prompt",
        data=scene_prompt.encode("utf-8"),
        original_name="scene_prompt.txt",
        mime="text/plain",
        kind="text",
    )
    ctx.record(
        name="base_scene",
        data=scene_bytes,
        original_name="base_scene.png",
        mime="image/png",
        kind="image",
    )

    # --- Stage 2: Initial composite (Gemini + Seedream Edit) ---
    sorted_product_ids = _sort_product_ids(product_image_content_ids, ctx.content_store)
    composite_prompt = _write_composite_prompt(
        scene_bytes, sorted_product_ids, ctx.content_store, gemini_model
    )
    logger.info("Composite prompt: %s", composite_prompt[:200])
    ctx.record(
        name="composite_prompt",
        data=composite_prompt.encode("utf-8"),
        original_name="composite_prompt.txt",
        mime="text/plain",
        kind="text",
    )

    product_urls = _upload_product_images(sorted_product_ids, ctx.content_store)
    scene_url = fal_client.upload(scene_bytes, content_type="image/png")

    composite_bytes = _run_edit(edit_model, composite_prompt, [scene_url, *product_urls])
    starting_frame_v0_id = ctx.record(
        name="starting_frame_v0",
        data=composite_bytes,
        original_name="starting_frame_v0.png",
        mime="image/png",
        kind="image",
    )
    logger.info("Initial composite recorded: %s", starting_frame_v0_id[:16])

    # --- Stage 3: Iterative refinement (Gemini critique loop) ---
    current_bytes = composite_bytes
    accepted = False
    last_critique: dict[str, Any] | None = None
    refinements_done = 0

    for i in range(1, max_refinements + 1):
        critique = _critique_composite(
            current_bytes, sorted_product_ids, ctx.content_store, gemini_model
        )
        logger.info("Refinement %d critique: %s", i, critique)

        ctx.record(
            name=f"critique_v{i}",
            data=json.dumps(critique, indent=2).encode("utf-8"),
            original_name=f"critique_v{i}.json",
            mime="application/json",
            kind="json",
        )
        last_critique = critique
        refinements_done = i - 1

        if critique.get("acceptable", False):
            accepted = True
            logger.info("Composite accepted after %d refinement(s)", i - 1)
            break

        correction = critique.get("correction_prompt", "")
        if not correction:
            logger.info("No correction prompt provided, stopping refinement")
            break

        logger.info("Refinement %d: %s", i, correction)

        current_url = fal_client.upload(current_bytes, content_type="image/png")
        current_bytes = _run_edit(edit_model, correction, [current_url])

        ctx.record(
            name=f"starting_frame_v{i}",
            data=current_bytes,
            original_name=f"starting_frame_v{i}.png",
            mime="image/png",
            kind="image",
        )
        refinements_done = i
    else:
        logger.warning("Reached max refinements (%d) without acceptance", max_refinements)

    final_id = ctx.record(
        name=OUTPUT_FINAL,
        data=current_bytes,
        original_name="starting_frame.png",
        mime="image/png",
        kind="image",
    )
    logger.info("Final starting frame recorded: %s", final_id[:16])

    ctx.set_attribute("accepted", accepted)
    ctx.set_attribute("refinements_done", refinements_done)
    ctx.set_attribute("max_refinements", max_refinements)
    if last_critique is not None:
        ctx.set_attribute("final_critique", last_critique)
    ctx.set_attribute("gemini_model", gemini_model)
    ctx.set_attribute("edit_model", edit_model)
    ctx.set_attribute("used_reference_video", reference_video_content_id is not None)
    ctx.set_attribute("used_reference_analysis", reference_analysis_content_id is not None)
    ctx.end()

    intermediate = {
        name: cid
        for name, cid in ctx.run_store.get(ctx.run_id).steps[ctx.step_name].outputs.items()
        if name != OUTPUT_FINAL
    }
    output_path = ctx.run_store.local_step_output_path(ctx.run_id, STEP_NAME, OUTPUT_FINAL)
    return StartingFrameResult(
        frame_path=output_path or Path(""),
        content_id=final_id,
        intermediate_content_ids=intermediate,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_base_scene(
    script: str,
    gemini_model: str,
    content_store: ContentStore,
    reference_video_content_id: str | None,
    reference_analysis: str | None,
) -> tuple[str, bytes]:
    """Have Gemini write a scene prompt grounded in the reference, then generate with Seedream.

    The Gemini call receives, in order:

    - the reference video file (uploaded to the Gemini File API, deleted after)
      when ``reference_video_content_id`` is set,
    - a user prompt built from ``scene_without_product.txt`` with ``$script``
      and ``$reference_analysis`` substituted.

    The returned ``scene_generation_prompt`` is then handed to Seedream
    text-to-image to produce the empty-ear base scene.
    """
    scene_template = load_prompt("scene_without_product", category="examples")
    scene_request = Template(scene_template).safe_substitute(
        script=script,
        reference_analysis=(reference_analysis or "(No reference analysis provided.)"),
    )

    client = get_gemini_client()
    contents: list[Any] = []

    uploaded_file_name: str | None = None
    tmpdir_ctx = tempfile.TemporaryDirectory()
    try:
        if reference_video_content_id is not None:
            video_path = content_store.materialize(
                reference_video_content_id, Path(tmpdir_ctx.name)
            )
            logger.info(
                "Uploading reference video for scene-prompt context (content=%s)...",
                reference_video_content_id[:16],
            )
            video_file = client.files.upload(file=str(video_path))
            uploaded_file_name = video_file.name
            if not uploaded_file_name:
                raise RuntimeError("Gemini File API did not return a file name")

            while video_file.state == "PROCESSING":
                time.sleep(FILE_POLL_INTERVAL_SECONDS)
                video_file = client.files.get(name=uploaded_file_name)
                logger.debug("Reference video file state: %s", video_file.state)
            if video_file.state != "ACTIVE":
                raise RuntimeError(f"Reference video processing failed (state={video_file.state})")
            contents.append(video_file)

        contents.append(scene_request)

        logger.info(
            "Generating scene prompt with Gemini %s (video=%s, analysis=%s)...",
            gemini_model,
            reference_video_content_id is not None,
            reference_analysis is not None,
        )
        response = client.models.generate_content(
            model=gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SCENE_PROMPT_SYSTEM,
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
    finally:
        if uploaded_file_name is not None:
            try:
                client.files.delete(name=uploaded_file_name)
                logger.debug("Cleaned up uploaded reference video: %s", uploaded_file_name)
            except (OSError, RuntimeError):
                logger.debug("Could not delete uploaded reference video (non-critical)")
        tmpdir_ctx.cleanup()

    scene_generation_prompt = response.text or ""
    if not scene_generation_prompt:
        raise RuntimeError("Gemini returned an empty scene prompt")
    logger.info("Scene prompt: %s", scene_generation_prompt[:120])

    logger.info("Generating base scene with %s...", BASE_SCENE_TEXT_TO_IMAGE)
    scene_result = fal_client.subscribe(
        BASE_SCENE_TEXT_TO_IMAGE,
        arguments={
            "prompt": scene_generation_prompt,
            "aspect_ratio": "3:4",
            "num_images": 1,
            "resolution": "1K",
            "output_format": "png",
        },
        with_logs=True,
    )
    scene_image_url = scene_result["images"][0]["url"]
    logger.info("Base scene URL: %s", scene_image_url)

    scene_response = httpx.get(scene_image_url)
    scene_response.raise_for_status()
    return scene_generation_prompt, bytes(scene_response.content)


def _sort_product_ids(content_ids: list[str], content_store: ContentStore) -> list[str]:
    """Sort product content ids so '_on' comes first, then alphabetically by original_name."""

    def key(content_id: str) -> tuple[int, str]:
        meta = content_store.get_meta(content_id)
        name = (meta.original_name or content_id).lower()
        return (0 if "_on" in name else 1, name)

    return sorted(content_ids, key=key)


def _mime_for(meta_mime: str | None, original_name: str | None) -> str:
    if meta_mime:
        return meta_mime
    if original_name:
        suffix = Path(original_name).suffix.lstrip(".").lower()
        if suffix:
            return f"image/{'jpeg' if suffix == 'jpg' else suffix}"
    return "image/png"


def _write_composite_prompt(
    scene_bytes: bytes,
    product_image_content_ids: list[str],
    content_store: ContentStore,
    gemini_model: str,
) -> str:
    """Send scene + all product images to Gemini and get a composite prompt."""
    prompt_text = load_prompt("write_composite_prompt", category="examples")

    content: list[types.Part | str] = [
        types.Part.from_bytes(data=scene_bytes, mime_type="image/png"),
    ]

    for content_id in product_image_content_ids:
        meta = content_store.get_meta(content_id)
        img_bytes = content_store.get_bytes(content_id)
        content.append(
            types.Part.from_bytes(
                data=img_bytes, mime_type=_mime_for(meta.mime, meta.original_name)
            )
        )

    content.append(prompt_text)

    client = get_gemini_client()
    logger.info(
        "Writing composite prompt with Gemini %s (scene + %d product images)...",
        gemini_model,
        len(product_image_content_ids),
    )
    response = client.models.generate_content(
        model=gemini_model,
        contents=content,
        config=types.GenerateContentConfig(
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
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
    product_image_content_ids: list[str],
    content_store: ContentStore,
    gemini_model: str,
) -> dict:
    """Send the composite + product references to Gemini for structured critique."""
    prompt_text = load_prompt("critique_composite", category="examples")

    content: list[types.Part | str] = [
        types.Part.from_bytes(data=composite_bytes, mime_type="image/png"),
    ]

    for content_id in product_image_content_ids:
        meta = content_store.get_meta(content_id)
        img_bytes = content_store.get_bytes(content_id)
        content.append(
            types.Part.from_bytes(
                data=img_bytes, mime_type=_mime_for(meta.mime, meta.original_name)
            )
        )

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
    """Run the edit model and return the result image bytes.

    Different fal endpoints use different parameters to control the output
    canvas: Seedream uses ``image_size`` (e.g. ``"portrait_4_3"``) while Nano
    Banana 2 uses ``aspect_ratio`` (e.g. ``"3:4"``). We branch on the model
    id so either family can be selected via ``--edit-model``.
    """
    logger.info("Running edit model %s...", edit_model)
    arguments: dict[str, Any] = {
        "prompt": prompt,
        "image_urls": image_urls,
        "num_images": 1,
    }
    if "nano-banana" in edit_model:
        arguments["aspect_ratio"] = "3:4"
        arguments["resolution"] = "1K"
        arguments["output_format"] = "png"
    else:
        arguments["image_size"] = "portrait_4_3"

    result = fal_client.subscribe(edit_model, arguments=arguments, with_logs=True)
    image_url = result["images"][0]["url"]
    response = httpx.get(image_url)
    response.raise_for_status()
    return bytes(response.content)


def _upload_product_images(content_ids: list[str], content_store: ContentStore) -> list[str]:
    """Upload all product images to fal and return URLs."""
    urls = []
    for content_id in content_ids:
        meta = content_store.get_meta(content_id)
        img_bytes = content_store.get_bytes(content_id)
        ct = _mime_for(meta.mime, meta.original_name)
        urls.append(fal_client.upload(img_bytes, content_type=ct))
        logger.info("  Uploaded product image: %s", meta.original_name or content_id[:16])
    return urls
