"""Step 3: Generate a starting frame via close-up-then-outpaint image generation.

Stage A:   Gemini writes a CLOSE-UP scene prompt (extreme 1:1 crop of the
           model's ear, no jewellery), Nano Banana 2 generates that close-up
           via text-to-image. The reference video and/or its analysis ground
           the lighting, skin tone, and styling.
Stage 1.5: Gemini independently captions every product reference photo.
           Each caption is content-tied (one Gemini call per image) and
           recorded as `product_captions.json`. Downstream Gemini calls
           refer to each photo by its caption rather than guessing roles
           from filename or multimodal ordering.
Stage B:   Gemini sees the captioned close-up + product photos and writes a
           composite prompt. The edit model composites the earring onto the
           close-up. A critique loop iterates against the captioned product
           references until the composite is accepted (or max refinements is
           hit). The composite stays at 1:1 throughout this stage.
Stage C:   A single deterministic outpaint expands the accepted close-up
           composite into the final 3:4 portrait that the video step
           consumes. Gemini writes the outpaint instruction, grounded in
           the reference video / analysis; the edit model then expands the
           canvas in one shot. There is no critique loop on this stage.
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

# Nano Banana 2 resolution for both base-scene generation and the edit loop.
# Supported: "0.5K", "1K" (fal default), "2K", "4K". A 4 mm earring on a
# portrait-aspect frame is only ~10 px tall at 1K — far below what the editor
# needs to represent the design correctly. 2K gives 4x the pixel budget at
# 1.5x the per-call cost ($0.058 -> ~$0.087); 4K is also available but slower
# and only worth it if 2K still proves insufficient.
NANO_BANANA_RESOLUTION = "2K"

STEP_NAME = "frame"
OUTPUT_FINAL = "starting_frame"

# Aspect ratios used per stage. Stage A and Stage B operate on the close-up
# (1:1, easier for the editor to size the earring relative to the lobe);
# Stage C outpaints to 3:4 to match the portrait framing the video step
# expects.
CLOSEUP_ASPECT_RATIO = "1:1"
FULL_FRAME_ASPECT_RATIO = "3:4"

SCENE_PROMPT_SYSTEM = (
    "You are an expert at writing prompts for AI image generation models. "
    "Write clear, detailed prompts that produce high-quality results. "
    "The scene MUST be an extreme close-up of the model's ear and the "
    "immediately surrounding skin (cheekbone, temple, side of jaw) — full "
    "face NOT in frame. The scene must NOT include any jewelry - it will "
    "be composited in later, and the canvas will be outpainted to a full "
    "portrait afterwards. When a reference video and/or its cinematography "
    "analysis are provided, match the reference's lighting style, skin tone, "
    "and model styling closely so the close-up can later be expanded into a "
    "consistent first frame.\n\n"
    "IMPORTANT: Avoid words that might trigger content filters like "
    '"naked", "bare", "exposed". Instead use words like "unadorned", '
    '"without jewelry", "empty earlobe".'
)

EXPAND_PROMPT_SYSTEM = (
    "You are an expert at writing prompts for AI image editing models. "
    "Your specific task is to write an outpainting (canvas-expansion) "
    "instruction that takes a tight 1:1 close-up of a model's ear (with the "
    "earring already correctly composited) and expands it into a 3:4 "
    "portrait of the same model. The single most important constraint is "
    "that everything visible in the input close-up — the ear, the earring, "
    "the surrounding skin, the lighting on those pixels — MUST be preserved "
    "EXACTLY. The editor must only invent the surrounding region (rest of "
    "head, hair, shoulders, background) so the final portrait matches the "
    "reference video's first frame in framing, lighting, pose, and wardrobe."
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
    A. Gemini writes a CLOSE-UP scene prompt grounded in the reference
       video / analysis (when provided), Nano Banana 2 generates a tight
       1:1 close-up of the model's ear WITHOUT the product.
    1.5. Gemini captions every unique product reference photo (results
       are content-tied and cached in the library across runs).
    B. Gemini writes a composite prompt; the edit model composites the
       earring onto the close-up. A critique loop runs against the
       captioned product references and corrects up to ``max_refinements``
       times. The composite stays at 1:1 throughout.
    C. A single deterministic outpaint expands the accepted close-up
       composite into the final 3:4 portrait. Gemini writes the outpaint
       instruction (using reference video / analysis as the framing
       target); the edit model then expands the canvas in one shot.

    All intermediates (close-up scene prompt, captions, composite prompt,
    every critique, every composite version, expand prompt, the final
    frame) are registered as content and attached to the step record so
    they can be inspected post-hoc.

    Args:
        script_content_id: Content id of the video script markdown.
        product_image_content_ids: Content ids of product images.
        ctx: Step context bound to the active run + step.
        gemini_model: Gemini model for scene + composite + critique + expand
            prompts.
        edit_model: Fal endpoint used for both Stage B (composite + critique
            corrections) and Stage C (outpaint). Defaults to
            ``fal-ai/nano-banana-2/edit``. The text-to-image variant used in
            Stage A is hard-coded to :data:`BASE_SCENE_TEXT_TO_IMAGE` and is
            intentionally decoupled from this flag.
        max_refinements: Maximum number of critique-and-correct iterations
            on the close-up composite (Stage B). The outpaint stage (Stage C)
            does not iterate.
        reference_video_content_id: Optional content id of the reference video.
            When set, the bytes are uploaded to the Gemini File API and passed
            to Stages A and C as additional cinematography context.
        reference_analysis_content_id: Optional content id of the reference
            analysis markdown. When set, the text is inlined into the Stage A
            and Stage C user prompts as additional context.

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

    # --- Stage A: Generate close-up scene (Gemini + Nano Banana 2 t2i) ---
    closeup_scene_prompt, closeup_scene_bytes = _generate_closeup_scene(
        script=script,
        gemini_model=gemini_model,
        content_store=ctx.content_store,
        reference_video_content_id=reference_video_content_id,
        reference_analysis=reference_analysis,
    )
    ctx.record(
        name="closeup_scene_prompt",
        data=closeup_scene_prompt.encode("utf-8"),
        original_name="closeup_scene_prompt.txt",
        mime="text/plain",
        kind="text",
    )
    ctx.record(
        name="closeup_scene",
        data=closeup_scene_bytes,
        original_name="closeup_scene.png",
        mime="image/png",
        kind="image",
    )

    # --- Stage 1.5: Resolve product photo captions (cached in the library) ---
    # Captions are persisted per-content-id in the content store, so once a
    # photo has been captioned (here or via scripts/precaption_products.py)
    # no future run will spend a Gemini call on it.
    sorted_product_ids = _sort_product_ids(product_image_content_ids, ctx.content_store)
    product_captions, captions_generated, captions_reused = ensure_product_captions(
        sorted_product_ids, ctx.content_store, gemini_model
    )
    captions_payload = [
        {"content_id": cid, "caption": product_captions[cid]} for cid in sorted_product_ids
    ]
    ctx.record(
        name="product_captions",
        data=json.dumps(captions_payload, indent=2).encode("utf-8"),
        original_name="product_captions.json",
        mime="application/json",
        kind="json",
    )
    logger.info(
        "Captions ready: %d generated, %d reused from library",
        captions_generated,
        captions_reused,
    )
    for cid, caption in product_captions.items():
        logger.info("  %s: %s", cid[:16], caption)

    # --- Stage B: Composite earring onto close-up + critique loop ---
    composite_prompt = _write_composite_prompt(
        closeup_scene_bytes,
        sorted_product_ids,
        product_captions,
        ctx.content_store,
        gemini_model,
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
    closeup_url = fal_client.upload(closeup_scene_bytes, content_type="image/png")

    composite_bytes = _run_edit(
        edit_model,
        composite_prompt,
        [closeup_url, *product_urls],
        aspect_ratio=CLOSEUP_ASPECT_RATIO,
    )
    closeup_v0_id = ctx.record(
        name="closeup_composite_v0",
        data=composite_bytes,
        original_name="closeup_composite_v0.png",
        mime="image/png",
        kind="image",
    )
    logger.info("Initial close-up composite recorded: %s", closeup_v0_id[:16])

    current_bytes = composite_bytes
    closeup_accepted = False
    last_critique: dict[str, Any] | None = None
    closeup_refinements_done = 0

    for i in range(1, max_refinements + 1):
        critique = _critique_composite(
            current_bytes,
            sorted_product_ids,
            product_captions,
            ctx.content_store,
            gemini_model,
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
        closeup_refinements_done = i - 1

        if critique.get("acceptable", False):
            closeup_accepted = True
            logger.info("Close-up composite accepted after %d refinement(s)", i - 1)
            break

        correction = critique.get("correction_prompt", "")
        if not correction:
            logger.info("No correction prompt provided, stopping refinement")
            break

        logger.info("Refinement %d: %s", i, correction)

        current_url = fal_client.upload(current_bytes, content_type="image/png")
        # Give the editor the same product references the initial composite
        # had. Without these, "make the earring 50% smaller" has no anchor
        # (the editor only sees the broken image and has no idea what the
        # correct size looks like). Order matters: the on-model photo is
        # the size ground truth, so it goes first after the current image,
        # matching the position the critique prompt tells the editor about.
        edit_image_urls = [current_url, *product_urls]
        current_bytes = _run_edit(
            edit_model,
            correction,
            edit_image_urls,
            aspect_ratio=CLOSEUP_ASPECT_RATIO,
        )

        ctx.record(
            name=f"closeup_composite_v{i}",
            data=current_bytes,
            original_name=f"closeup_composite_v{i}.png",
            mime="image/png",
            kind="image",
        )
        closeup_refinements_done = i
    else:
        logger.warning("Reached max refinements (%d) without acceptance", max_refinements)

    closeup_final_id = ctx.record(
        name="closeup_composite_final",
        data=current_bytes,
        original_name="closeup_composite_final.png",
        mime="image/png",
        kind="image",
    )
    logger.info("Final close-up composite recorded: %s", closeup_final_id[:16])

    # --- Stage C: Outpaint close-up to full 3:4 portrait (single shot) ---
    expand_prompt, full_frame_bytes = _expand_to_full_frame(
        closeup_composite_bytes=current_bytes,
        gemini_model=gemini_model,
        edit_model=edit_model,
        content_store=ctx.content_store,
        reference_video_content_id=reference_video_content_id,
        reference_analysis=reference_analysis,
    )
    ctx.record(
        name="expand_prompt",
        data=expand_prompt.encode("utf-8"),
        original_name="expand_prompt.txt",
        mime="text/plain",
        kind="text",
    )

    final_id = ctx.record(
        name=OUTPUT_FINAL,
        data=full_frame_bytes,
        original_name="starting_frame.png",
        mime="image/png",
        kind="image",
    )
    logger.info("Final starting frame (outpainted) recorded: %s", final_id[:16])

    ctx.set_attribute("closeup_accepted", closeup_accepted)
    ctx.set_attribute("closeup_refinements_done", closeup_refinements_done)
    ctx.set_attribute("max_refinements", max_refinements)
    ctx.set_attribute("outpaint_done", True)
    if last_critique is not None:
        ctx.set_attribute("final_critique", last_critique)
    ctx.set_attribute("gemini_model", gemini_model)
    ctx.set_attribute("edit_model", edit_model)
    ctx.set_attribute("base_scene_text_to_image_model", BASE_SCENE_TEXT_TO_IMAGE)
    ctx.set_attribute("used_reference_video", reference_video_content_id is not None)
    ctx.set_attribute("used_reference_analysis", reference_analysis_content_id is not None)
    ctx.set_attribute("captions_generated", captions_generated)
    ctx.set_attribute("captions_reused", captions_reused)
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


def _generate_closeup_scene(
    script: str,
    gemini_model: str,
    content_store: ContentStore,
    reference_video_content_id: str | None,
    reference_analysis: str | None,
) -> tuple[str, bytes]:
    """Have Gemini write a CLOSE-UP scene prompt, then generate it with Nano Banana 2.

    The Gemini call receives, in order:

    - the reference video file (uploaded to the Gemini File API, deleted after)
      when ``reference_video_content_id`` is set,
    - a user prompt built from ``closeup_scene_without_product.txt`` with
      ``$script`` and ``$reference_analysis`` substituted.

    The returned ``closeup_scene_prompt`` is then handed to Nano Banana 2
    text-to-image to produce a tight 1:1 close-up of the ear/cheek with no
    jewellery — the canvas the editor will composite the earring onto in
    Stage B.
    """
    scene_template = load_prompt("closeup_scene_without_product", category="examples")
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
                "Uploading reference video for close-up scene context (content=%s)...",
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
            "Generating close-up scene prompt with Gemini %s (video=%s, analysis=%s)...",
            gemini_model,
            reference_video_content_id is not None,
            reference_analysis is not None,
        )
        response = client.models.generate_content(
            model=gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SCENE_PROMPT_SYSTEM,
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

    closeup_scene_prompt = response.text or ""
    if not closeup_scene_prompt:
        raise RuntimeError("Gemini returned an empty close-up scene prompt")
    logger.info("Close-up scene prompt: %s", closeup_scene_prompt[:120])

    logger.info(
        "Generating close-up scene with %s (aspect=%s, resolution=%s)...",
        BASE_SCENE_TEXT_TO_IMAGE,
        CLOSEUP_ASPECT_RATIO,
        NANO_BANANA_RESOLUTION,
    )
    scene_result = fal_client.subscribe(
        BASE_SCENE_TEXT_TO_IMAGE,
        arguments={
            "prompt": closeup_scene_prompt,
            "aspect_ratio": CLOSEUP_ASPECT_RATIO,
            "num_images": 1,
            "resolution": NANO_BANANA_RESOLUTION,
            "output_format": "png",
        },
        with_logs=True,
    )
    scene_image_url = scene_result["images"][0]["url"]
    logger.info("Close-up scene URL: %s", scene_image_url)

    scene_response = httpx.get(scene_image_url)
    scene_response.raise_for_status()
    return closeup_scene_prompt, bytes(scene_response.content)


def _expand_to_full_frame(
    closeup_composite_bytes: bytes,
    gemini_model: str,
    edit_model: str,
    content_store: ContentStore,
    reference_video_content_id: str | None,
    reference_analysis: str | None,
) -> tuple[str, bytes]:
    """Outpaint a 1:1 close-up composite into a 3:4 portrait, in one shot.

    Two-step internal flow:

    1. Gemini sees the close-up composite (Photo 1) and, when available, the
       reference video / analysis as the framing target. It writes a single
       outpainting instruction grounded in
       ``expand_to_full_frame.txt`` — telling the editor to PRESERVE the
       close-up region exactly and INVENT only the surrounding head, hair,
       shoulders, and background to match the reference.
    2. The edit model receives the close-up composite as its sole input
       image alongside the outpaint instruction and produces the final
       3:4 portrait. There is no critique loop on this stage.
    """
    expand_template = load_prompt("expand_to_full_frame", category="examples")
    expand_request = Template(expand_template).safe_substitute(
        reference_analysis=(reference_analysis or "(No reference analysis provided.)"),
    )

    client = get_gemini_client()
    contents: list[Any] = [
        "Photo 1 (CLOSE-UP COMPOSITE — must be preserved exactly):",
        types.Part.from_bytes(data=closeup_composite_bytes, mime_type="image/png"),
    ]

    uploaded_file_name: str | None = None
    tmpdir_ctx = tempfile.TemporaryDirectory()
    try:
        if reference_video_content_id is not None:
            video_path = content_store.materialize(
                reference_video_content_id, Path(tmpdir_ctx.name)
            )
            logger.info(
                "Uploading reference video for expand-prompt context (content=%s)...",
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

        contents.append(expand_request)

        logger.info(
            "Writing expand prompt with Gemini %s (video=%s, analysis=%s)...",
            gemini_model,
            reference_video_content_id is not None,
            reference_analysis is not None,
        )
        response = client.models.generate_content(
            model=gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=EXPAND_PROMPT_SYSTEM,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
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

    expand_prompt = response.text or ""
    if not expand_prompt:
        raise RuntimeError("Gemini returned an empty expand prompt")
    logger.info("Expand prompt: %s", expand_prompt[:200])

    closeup_url = fal_client.upload(closeup_composite_bytes, content_type="image/png")
    full_frame_bytes = _run_edit(
        edit_model,
        expand_prompt,
        [closeup_url],
        aspect_ratio=FULL_FRAME_ASPECT_RATIO,
    )
    return expand_prompt, full_frame_bytes


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


def _gemini_caption_one_image(
    content_id: str,
    content_store: ContentStore,
    gemini_model: str,
    prompt_text: str,
    client: Any,
) -> str:
    """Single Gemini call to caption one image. Raises on empty output."""
    meta = content_store.get_meta(content_id)
    img_bytes = content_store.get_bytes(content_id)
    mime = _mime_for(meta.mime, meta.original_name)

    response = client.models.generate_content(
        model=gemini_model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime),
            prompt_text,
        ],
        config=types.GenerateContentConfig(
            # The captioning prompt itself constrains the output to a single
            # short sentence (max 30 words); we let the prompt do the work
            # rather than hard-clamping the token budget.
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        ),
    )
    caption = (response.text or "").strip()
    if not caption:
        raise RuntimeError(
            f"Gemini returned an empty caption for product image {content_id[:16]} "
            f"({meta.original_name or '<unknown>'})"
        )
    return caption


def ensure_product_captions(
    product_image_content_ids: list[str],
    content_store: ContentStore,
    gemini_model: str,
) -> tuple[dict[str, str], int, int]:
    """Return ``{content_id: caption}`` for all product images, lazily generating.

    For each unique ``content_id``:

    1. If the content store already holds a caption sidecar
       (``library/<aa>/<sha>.caption.json``), reuse it verbatim. No LLM call.
    2. Otherwise, ask Gemini once with a strict captioning prompt, then
       persist the result back to the content store via
       :meth:`ContentStore.set_caption` so every future run reuses it.

    Returns a tuple ``(captions, generated_count, reused_count)`` where
    ``captions`` covers every input id (deduped on identical content). A
    properly seeded library will have ``generated_count == 0``.
    """
    prompt_text = load_prompt("caption_product_image", category="examples")
    client: Any | None = None

    captions: dict[str, str] = {}
    generated = 0
    reused = 0
    for content_id in product_image_content_ids:
        if content_id in captions:
            continue

        existing = content_store.get_caption(content_id)
        if existing is not None:
            captions[content_id] = existing
            reused += 1
            continue

        if client is None:
            client = get_gemini_client()
        caption = _gemini_caption_one_image(
            content_id=content_id,
            content_store=content_store,
            gemini_model=gemini_model,
            prompt_text=prompt_text,
            client=client,
        )
        content_store.set_caption(content_id, caption, model=gemini_model)
        captions[content_id] = caption
        generated += 1

    return captions, generated, reused


def _build_captioned_contents(
    leading_image_caption: str,
    leading_image_bytes: bytes,
    leading_image_mime: str,
    product_image_content_ids: list[str],
    product_captions: dict[str, str],
    content_store: ContentStore,
    trailing_text: str,
) -> list[types.Part | str]:
    """Build a multimodal `contents` list with explicit per-image captions.

    Layout::

        [
            "Photo 1 (<leading_image_caption>):",
            <leading image part>,
            "Photo 2 (<gemini-generated caption>):",
            <product image part>,
            ...,
            <trailing_text>,
        ]

    ``product_captions`` is the mapping returned by
    :func:`_caption_product_images`; its keys must include every entry of
    ``product_image_content_ids``.
    """
    parts: list[types.Part | str] = [
        f"Photo 1 ({leading_image_caption}):",
        types.Part.from_bytes(data=leading_image_bytes, mime_type=leading_image_mime),
    ]
    for idx, content_id in enumerate(product_image_content_ids, start=2):
        meta = content_store.get_meta(content_id)
        try:
            caption = product_captions[content_id]
        except KeyError as exc:
            raise KeyError(
                f"Missing caption for product image content_id={content_id[:16]}"
            ) from exc
        img_bytes = content_store.get_bytes(content_id)
        parts.append(f"Photo {idx} ({caption}):")
        parts.append(
            types.Part.from_bytes(
                data=img_bytes, mime_type=_mime_for(meta.mime, meta.original_name)
            )
        )
    parts.append(trailing_text)
    return parts


def _write_composite_prompt(
    scene_bytes: bytes,
    product_image_content_ids: list[str],
    product_captions: dict[str, str],
    content_store: ContentStore,
    gemini_model: str,
) -> str:
    """Send scene + all product images to Gemini and get a composite prompt.

    Each image is preceded by an explicit caption — ``Photo 1 (BASE SCENE...)``
    for the scene, and a Gemini-generated content-tied caption (e.g.
    ``Photo 2 (ON A REAL MODEL'S EAR — ...)``) for each product photo — so
    Gemini does not have to infer image roles from ordering or filenames.
    Thinking is set to HIGH because the visual analysis (size estimation,
    design transcription) is exactly what produces a high-fidelity edit prompt.
    """
    prompt_text = load_prompt("write_composite_prompt", category="examples")

    content = _build_captioned_contents(
        leading_image_caption="BASE SCENE \u2014 model without any jewelry",
        leading_image_bytes=scene_bytes,
        leading_image_mime="image/png",
        product_image_content_ids=product_image_content_ids,
        product_captions=product_captions,
        content_store=content_store,
        trailing_text=prompt_text,
    )

    client = get_gemini_client()
    logger.info(
        "Writing composite prompt with Gemini %s (scene + %d captioned product images, thinking=HIGH)...",
        gemini_model,
        len(product_image_content_ids),
    )
    response = client.models.generate_content(
        model=gemini_model,
        contents=content,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
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
    product_captions: dict[str, str],
    content_store: ContentStore,
    gemini_model: str,
) -> dict:
    """Send the composite + product references to Gemini for structured critique."""
    prompt_text = load_prompt("critique_composite", category="examples")

    content = _build_captioned_contents(
        leading_image_caption="CURRENT COMPOSITE \u2014 the image you are critiquing",
        leading_image_bytes=composite_bytes,
        leading_image_mime="image/png",
        product_image_content_ids=product_image_content_ids,
        product_captions=product_captions,
        content_store=content_store,
        trailing_text=prompt_text,
    )

    client = get_gemini_client()
    logger.info("Critiquing composite with Gemini %s (structured output)...", gemini_model)
    response = client.models.generate_content(
        model=gemini_model,
        contents=content,
        config=types.GenerateContentConfig(
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


# Per-endpoint mapping from generic ``aspect_ratio`` (one of "1:1", "3:4")
# to the enum string the endpoint actually expects. NB2 and FLUX Kontext
# accept the generic ratios directly. GPT Image and Seedream want a fixed
# size enum.
_GPT_IMAGE_SIZE = {"1:1": "1024x1024", "3:4": "1024x1536"}
_SEEDREAM_IMAGE_SIZE = {"1:1": "square_hd", "3:4": "portrait_4_3"}


def _run_edit(
    edit_model: str,
    prompt: str,
    image_urls: list[str],
    aspect_ratio: str = FULL_FRAME_ASPECT_RATIO,
) -> bytes:
    """Run the edit model and return the result image bytes.

    Different fal endpoints use different schemas. We branch on the model id
    so any of the supported families can be selected via ``--edit-model``:

    - ``fal-ai/nano-banana-2/edit`` (Google, Gemini 3.1 Flash Image): takes
      ``aspect_ratio`` + ``resolution`` and accepts up to 14 reference images
      via ``image_urls``.
    - ``fal-ai/gpt-image-1.5/edit`` and ``fal-ai/gpt-image-1/edit-image``
      (OpenAI): take ``image_size`` as a fixed enum (``"1024x1024"`` for
      square, ``"1024x1536"`` for portrait), plus ``quality`` and
      ``input_fidelity`` knobs. Multi-image input via ``image_urls``.
    - ``fal-ai/flux-pro/kontext/multi`` (Black Forest Labs): the experimental
      multi-reference Kontext endpoint. Takes ``aspect_ratio`` + ``image_urls``.
    - everything else (Seedream variants): ``image_size`` enum
      (``"square_hd"`` for 1:1, ``"portrait_4_3"`` for 3:4).

    ``aspect_ratio`` is a stage-level concept rather than a per-endpoint
    one, so callers pass the generic ratio (``"1:1"`` for the close-up
    composite + critique loop, ``"3:4"`` for the final outpaint) and
    this function translates to whatever the endpoint expects.
    """
    logger.info("Running edit model %s (aspect=%s)...", edit_model, aspect_ratio)
    arguments: dict[str, Any] = {
        "prompt": prompt,
        "image_urls": image_urls,
        "num_images": 1,
    }
    if "nano-banana" in edit_model:
        arguments["aspect_ratio"] = aspect_ratio
        arguments["resolution"] = NANO_BANANA_RESOLUTION
        arguments["output_format"] = "png"
    elif "gpt-image" in edit_model:
        arguments["image_size"] = _GPT_IMAGE_SIZE.get(aspect_ratio, "1024x1536")
        arguments["quality"] = "high"
        arguments["input_fidelity"] = "high"
        arguments["output_format"] = "png"
    elif "flux-pro/kontext" in edit_model:
        arguments["aspect_ratio"] = aspect_ratio
        arguments["output_format"] = "png"
        arguments["safety_tolerance"] = "6"
        if len(image_urls) > 4:
            logger.info(
                "FLUX Kontext multi caps image_urls at 4; truncating from %d to 4 "
                "(keeping current composite + first 3 references).",
                len(image_urls),
            )
            arguments["image_urls"] = image_urls[:4]
    else:
        arguments["image_size"] = _SEEDREAM_IMAGE_SIZE.get(aspect_ratio, "portrait_4_3")

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
