"""Pipeline orchestrator for end-to-end video generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from video_generation.clients import load_env
from video_generation.config import (
    PipelineConfig,
    PipelineResult,
)
from video_generation.steps.analyze_reference import analyze_reference
from video_generation.steps.generate_starting_frame import generate_starting_frame
from video_generation.steps.generate_video import generate_video
from video_generation.steps.write_script import discover_product_images, write_script
from video_generation.store import (
    ContentStore,
    RunInputs,
    RunParams,
    RunRecord,
    RunStore,
    StepContext,
    current_code_version,
    make_storage,
)
from video_generation.store.models import (
    STEP_ANALYZE,
    STEP_FRAME,
    STEP_SCRIPT,
    STEP_VIDEO,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------


def _build_stores(config: PipelineConfig) -> tuple[ContentStore, RunStore]:
    storage = make_storage(config.storage_spec)
    content_store = ContentStore(storage)
    run_store = RunStore(storage, content_store)
    return content_store, run_store


def _params_from_config(config: PipelineConfig) -> RunParams:
    return RunParams(
        claude_model=config.claude_model,
        gemini_model=config.gemini_model,
        edit_model=config.edit_model,
        video_model=config.video_model,
        video_duration=config.video_duration,
        max_refinements=config.max_refinements,
        variant=config.variant,
    )


def _register_inputs(config: PipelineConfig, content_store: ContentStore) -> RunInputs:
    """Register inputs from the CLI-style config and return a :class:`RunInputs`."""
    product_paths = discover_product_images(config.product_dir)
    product_ids = [content_store.register_path(p, kind="image").content_id for p in product_paths]

    reference_video_id: str | None = None
    if config.reference_video:
        reference_video_id = content_store.register_path(
            config.reference_video, kind="video"
        ).content_id

    reference_analysis_id: str | None = None
    if config.reference_analysis:
        reference_analysis_id = content_store.register_path(
            config.reference_analysis, kind="text"
        ).content_id

    return RunInputs(
        product_images=product_ids,
        reference_video=reference_video_id,
        reference_analysis=reference_analysis_id,
    )


def _make_step_ctx(
    run_store: RunStore,
    content_store: ContentStore,
    run_id: str,
    step_name: str,
    inputs: dict,
) -> StepContext:
    return StepContext(
        content_store=content_store,
        run_store=run_store,
        run_id=run_id,
        step_name=step_name,
        inputs=inputs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full video generation pipeline.

    Steps:
        1. Analyze reference video (or use existing analysis)
        2. Write a video script
        3. Generate a starting frame image
        4. Generate the final video

    The run is identified by a deterministic ``run_id`` derived from
    ``(inputs, params, code_version)``. Re-invoking with the same inputs is
    idempotent: completed steps are skipped, partial work resumes.

    Args:
        config: Pipeline configuration.

    Returns:
        PipelineResult with paths and content ids for every artifact.
    """
    load_env()
    config.validate()

    content_store, run_store = _build_stores(config)
    params = _params_from_config(config)
    inputs = _register_inputs(config, content_store)
    code_version = current_code_version()

    run = run_store.create_or_load(inputs=inputs, params=params, code_version=code_version)
    logger.info(
        "Run id: %s (git=%s, dirty=%s)",
        run.run_id,
        code_version.git_sha[:8],
        code_version.git_dirty,
    )

    # --- Step 1: Analyze reference (Gemini - native video understanding) ---
    analysis_id = _ensure_analysis(run, run_store, content_store, config)

    # --- Step 2: Write script ---
    script_id = _ensure_script(run, run_store, content_store, config, analysis_id)

    # --- Step 3: Generate starting frame ---
    frame_id = _ensure_frame(run, run_store, content_store, config, script_id, analysis_id)

    # --- Step 4: Generate video ---
    video_id = _ensure_video(run, run_store, content_store, config, frame_id, script_id)

    final_run = run_store.mark_finished(run.run_id)

    result = PipelineResult(
        analysis_path=run_store.local_step_output_path(run.run_id, STEP_ANALYZE, "analysis")
        or Path(""),
        script_path=run_store.local_step_output_path(run.run_id, STEP_SCRIPT, "script") or Path(""),
        starting_frame_path=run_store.local_step_output_path(
            run.run_id, STEP_FRAME, "starting_frame"
        )
        or Path(""),
        video_path=run_store.local_step_output_path(run.run_id, STEP_VIDEO, "video") or Path(""),
        run_id=final_run.run_id,
        analysis_content_id=analysis_id,
        script_content_id=script_id,
        starting_frame_content_id=frame_id,
        video_content_id=video_id,
    )
    logger.info(result.summary())
    return result


def run_step(step_name: str, config: PipelineConfig) -> None:
    """Run a single pipeline step within a (resumable) run."""
    load_env()
    content_store, run_store = _build_stores(config)
    params = _params_from_config(config)
    inputs = _register_inputs(config, content_store)
    code_version = current_code_version()
    run = run_store.create_or_load(inputs=inputs, params=params, code_version=code_version)

    if step_name == STEP_ANALYZE:
        if not config.reference_video:
            raise ValueError("--reference-video is required for the 'analyze' step")
        _ensure_analysis(run, run_store, content_store, config, force=True)

    elif step_name == STEP_SCRIPT:
        analysis_id = _ensure_analysis(run, run_store, content_store, config)
        _ensure_script(run, run_store, content_store, config, analysis_id, force=True)

    elif step_name == STEP_FRAME:
        analysis_id = _ensure_analysis(run, run_store, content_store, config)
        script_id = _ensure_script(run, run_store, content_store, config, analysis_id)
        _ensure_frame(run, run_store, content_store, config, script_id, analysis_id, force=True)

    elif step_name == STEP_VIDEO:
        analysis_id = _ensure_analysis(run, run_store, content_store, config)
        script_id = _ensure_script(run, run_store, content_store, config, analysis_id)
        frame_id = _ensure_frame(run, run_store, content_store, config, script_id, analysis_id)
        _ensure_video(run, run_store, content_store, config, frame_id, script_id, force=True)

    else:
        raise ValueError(
            f"Unknown step: {step_name!r}. Must be one of: "
            f"{STEP_ANALYZE}, {STEP_SCRIPT}, {STEP_FRAME}, {STEP_VIDEO}"
        )


# ---------------------------------------------------------------------------
# Per-step idempotent runners
# ---------------------------------------------------------------------------


def _completed_output_id(run: RunRecord, step_name: str, output_name: str) -> str | None:
    """Return the content id if the step is completed and produced ``output_name``."""
    step = run.steps.get(step_name)
    if step is None or not step.is_completed:
        return None
    return step.outputs.get(output_name)


def _ensure_analysis(
    run: RunRecord,
    run_store: RunStore,
    content_store: ContentStore,
    config: PipelineConfig,
    force: bool = False,
) -> str:
    """Return the content id for the reference analysis, running step 1 if needed."""
    if not force:
        existing = _completed_output_id(run, STEP_ANALYZE, "analysis")
        if existing is not None:
            logger.info("Step 1 cached: %s", existing[:16])
            return existing
        if run.inputs.reference_analysis is not None:
            logger.info(
                "Step 1 skipped: using provided analysis %s",
                run.inputs.reference_analysis[:16],
            )
            return run.inputs.reference_analysis

    if not run.inputs.reference_video:
        if run.inputs.reference_analysis is None:
            raise ValueError("Either reference_video or reference_analysis is required")
        return run.inputs.reference_analysis

    logger.info("=== Step 1: Analyze Reference Video ===")
    ctx = _make_step_ctx(
        run_store,
        content_store,
        run.run_id,
        STEP_ANALYZE,
        inputs={"reference_video": run.inputs.reference_video},
    )
    result = analyze_reference(
        run.inputs.reference_video,
        ctx=ctx,
        gemini_model=config.gemini_model,
        num_frames=config.num_frames,
    )
    assert result.content_id is not None
    return result.content_id


def _ensure_script(
    run: RunRecord,
    run_store: RunStore,
    content_store: ContentStore,
    config: PipelineConfig,
    analysis_id: str,
    force: bool = False,
) -> str:
    if not force:
        existing = _completed_output_id(run, STEP_SCRIPT, "script")
        if existing is not None:
            logger.info("Step 2 cached: %s", existing[:16])
            return existing

    logger.info("=== Step 2: Write Script ===")
    ctx = _make_step_ctx(
        run_store,
        content_store,
        run.run_id,
        STEP_SCRIPT,
        inputs={
            "reference_analysis": analysis_id,
            "product_images": list(run.inputs.product_images),
        },
    )
    result = write_script(
        analysis_id,
        list(run.inputs.product_images),
        ctx=ctx,
        gemini_model=config.gemini_model,
    )
    assert result.content_id is not None
    return result.content_id


def _ensure_frame(
    run: RunRecord,
    run_store: RunStore,
    content_store: ContentStore,
    config: PipelineConfig,
    script_id: str,
    analysis_id: str | None,
    force: bool = False,
) -> str:
    if not force:
        existing = _completed_output_id(run, STEP_FRAME, "starting_frame")
        if existing is not None:
            logger.info("Step 3 cached: %s", existing[:16])
            return existing

    logger.info("=== Step 3: Generate Starting Frame ===")
    step_inputs: dict[str, Any] = {
        "script": script_id,
        "product_images": list(run.inputs.product_images),
    }
    if run.inputs.reference_video is not None:
        step_inputs["reference_video"] = run.inputs.reference_video
    if analysis_id is not None:
        step_inputs["reference_analysis"] = analysis_id

    ctx = _make_step_ctx(
        run_store,
        content_store,
        run.run_id,
        STEP_FRAME,
        inputs=step_inputs,
    )
    result = generate_starting_frame(
        script_id,
        list(run.inputs.product_images),
        ctx=ctx,
        gemini_model=config.gemini_model,
        edit_model=config.edit_model,
        max_refinements=config.max_refinements,
        reference_video_content_id=run.inputs.reference_video,
        reference_analysis_content_id=analysis_id,
    )
    assert result.content_id is not None
    return result.content_id


def _ensure_video(
    run: RunRecord,
    run_store: RunStore,
    content_store: ContentStore,
    config: PipelineConfig,
    frame_id: str,
    script_id: str,
    force: bool = False,
) -> str:
    if not force:
        existing = _completed_output_id(run, STEP_VIDEO, "video")
        if existing is not None:
            logger.info("Step 4 cached: %s", existing[:16])
            return existing

    logger.info("=== Step 4: Generate Video ===")
    ctx = _make_step_ctx(
        run_store,
        content_store,
        run.run_id,
        STEP_VIDEO,
        inputs={"starting_frame": frame_id, "script": script_id},
    )
    result = generate_video(
        frame_id,
        script_id,
        ctx=ctx,
        claude_model=config.claude_model,
        video_model=config.video_model,
        duration=config.video_duration,
    )
    assert result.content_id is not None
    return result.content_id
