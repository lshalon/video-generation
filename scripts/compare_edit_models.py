"""Run the same critique correction across multiple image-editing models.

Given a previous pipeline run, this script:

1. Reads the run's manifest.
2. Pulls the initial composite image (``starting_frame_v0`` by default) and
   the correction prompt produced by a chosen critique iteration
   (``critique_v1`` by default) out of the content store.
3. Uploads the composite + the original product reference photos to fal in
   the same ``[current_composite, *product_urls]`` order the live pipeline
   uses during refinement.
4. Calls each candidate edit model with the *same* prompt + *same* inputs
   and writes the outputs into ``data/comparisons/<run_id>/<model_slug>.png``
   so they can be opened side-by-side.

The point is an apples-to-apples test of which editor actually shrinks the
earring (or whatever the critique asked for) — isolating the editor from
prompt quality and reference selection, both of which are held constant.

Run with::

    uv run python scripts/compare_edit_models.py --run-id 10b8dc902fa2ab08

Defaults to the most recently modified run under ``data/runs/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fal_client

from video_generation.clients.config import get_api_key, load_env
from video_generation.steps.generate_starting_frame import (
    _mime_for,
    _run_edit,
    _sort_product_ids,
)
from video_generation.store import ContentStore, LocalStorage, RunStore, make_storage

logger = logging.getLogger(__name__)

DEFAULT_EDIT_MODELS: list[str] = [
    "fal-ai/nano-banana-2/edit",
    "fal-ai/gpt-image-1.5/edit",
    "fal-ai/flux-pro/kontext/multi",
    "fal-ai/bytedance/seedream/v4.5/edit",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id to source the composite and critique from (default: latest run).",
    )
    parser.add_argument(
        "--composite",
        type=str,
        default="closeup_composite_v0",
        help=(
            "Step output name to use as the input composite (default: "
            "closeup_composite_v0 — the pre-refinement initial close-up "
            "composite from the new pipeline). For runs predating the "
            "close-up-then-outpaint refactor, pass --composite starting_frame_v0."
        ),
    )
    parser.add_argument(
        "--critique",
        type=str,
        default="critique_v1",
        help=(
            "Step output name of the critique JSON to source the correction prompt from "
            "(default: critique_v1)."
        ),
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="local:./data",
        help="Storage spec for the content store (default: local:./data).",
    )
    parser.add_argument(
        "--edit-models",
        type=str,
        nargs="+",
        default=DEFAULT_EDIT_MODELS,
        help=(
            "Edit model endpoints to compare. Defaults to nano-banana-2, "
            "gpt-image-1.5, flux-pro/kontext/multi, and seedream v4.5."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write the comparison outputs into "
            "(default: data/comparisons/<run_id>/)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel fal calls. Each model is independent so the default of 4 is safe.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="1:1",
        choices=("1:1", "3:4"),
        help=(
            "Output aspect ratio passed to each editor. Default '1:1' "
            "matches the new close-up composite. Use '3:4' for old "
            "starting_frame_v0 sources or to evaluate outpainting behaviour."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _resolve_run_id(run_store: RunStore, requested: str | None) -> str:
    if requested:
        return requested
    if not isinstance(run_store.storage, LocalStorage):
        raise RuntimeError("RunStore is not backed by local storage; pass --run-id explicitly.")
    runs_root = run_store.storage.root / "runs"
    if not runs_root.is_dir():
        raise RuntimeError(f"No runs directory at {runs_root}")
    candidates = [p for p in runs_root.iterdir() if (p / "manifest.json").exists()]
    if not candidates:
        raise RuntimeError(f"No runs found under {runs_root}")
    latest = max(candidates, key=lambda p: (p / "manifest.json").stat().st_mtime)
    return latest.name


def _slugify(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _run_one(
    model: str,
    correction_prompt: str,
    edit_image_urls: list[str],
    out_path: Path,
    aspect_ratio: str,
) -> tuple[str, Path, str | None]:
    try:
        logger.info("[%s] running (aspect=%s)...", model, aspect_ratio)
        result = _run_edit(model, correction_prompt, edit_image_urls, aspect_ratio=aspect_ratio)
        out_path.write_bytes(result)
        logger.info("[%s] wrote %s (%d bytes)", model, out_path, len(result))
        return model, out_path, None
    except Exception as exc:  # noqa: BLE001 - intentionally broad: one model's failure
        # must not abort the comparison; record the error and continue with the rest.
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[%s] FAILED: %s", model, msg)
        return model, out_path, msg


def compare(
    run_id: str | None,
    composite_name: str,
    critique_name: str,
    storage_spec: str,
    edit_models: list[str],
    out_dir: Path | None,
    max_workers: int,
    aspect_ratio: str,
) -> None:
    load_env()
    get_api_key("fal")  # fail fast with a clear message if FAL_KEY is missing

    storage = make_storage(storage_spec)
    content_store = ContentStore(storage)
    run_store = RunStore(storage, content_store)

    resolved_run_id = _resolve_run_id(run_store, run_id)
    record = run_store.get(resolved_run_id)
    logger.info("Using run %s (status=%s)", resolved_run_id, record.status)

    frame_step = record.steps.get("frame")
    if frame_step is None:
        raise RuntimeError(f"Run {resolved_run_id} has no 'frame' step.")

    if composite_name not in frame_step.outputs:
        raise KeyError(
            f"'{composite_name}' not found in frame outputs. Available: "
            f"{sorted(frame_step.outputs)}"
        )
    if critique_name not in frame_step.outputs:
        raise KeyError(
            f"'{critique_name}' not found in frame outputs. Available: "
            f"{sorted(frame_step.outputs)}"
        )

    composite_id = frame_step.outputs[composite_name]
    critique_id = frame_step.outputs[critique_name]

    critique = json.loads(content_store.get_bytes(critique_id).decode("utf-8"))
    correction_prompt = critique.get("correction_prompt", "")
    if not correction_prompt:
        raise RuntimeError(
            f"Critique {critique_name} has no correction_prompt — "
            "looks like that iteration accepted the composite or returned empty."
        )

    product_ids = frame_step.inputs.get("product_images")
    if not product_ids:
        raise RuntimeError(f"Run {resolved_run_id} frame step has no product_images inputs.")
    sorted_product_ids = _sort_product_ids(product_ids, content_store)

    logger.info("Composite source: %s (%s)", composite_name, composite_id[:16])
    logger.info("Critique source:  %s (%s)", critique_name, critique_id[:16])
    logger.info(
        "Acceptable=%s, issues=%d", critique.get("acceptable"), len(critique.get("issues", []))
    )
    logger.info("Correction prompt:\n%s\n", correction_prompt)

    composite_bytes = content_store.get_bytes(composite_id)
    composite_url = fal_client.upload(composite_bytes, content_type="image/png")
    logger.info("Uploaded composite to fal.")

    product_urls: list[str] = []
    for cid in sorted_product_ids:
        meta = content_store.get_meta(cid)
        img_bytes = content_store.get_bytes(cid)
        ct = _mime_for(meta.mime, meta.original_name)
        product_urls.append(fal_client.upload(img_bytes, content_type=ct))
        logger.info("  Uploaded reference: %s (%s)", meta.original_name or cid[:16], ct)

    edit_image_urls = [composite_url, *product_urls]
    logger.info("Sending %d images to each editor.", len(edit_image_urls))

    if out_dir is None:
        out_dir = Path("data/comparisons") / resolved_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "correction_prompt.txt").write_text(correction_prompt, encoding="utf-8")
    (out_dir / "source_composite.png").write_bytes(composite_bytes)
    metadata = {
        "run_id": resolved_run_id,
        "composite_source": {"name": composite_name, "content_id": composite_id},
        "critique_source": {"name": critique_name, "content_id": critique_id},
        "critique": critique,
        "edit_models": edit_models,
        "image_input_order": [composite_id, *sorted_product_ids],
    }
    (out_dir / "comparison_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results: list[tuple[str, Path, str | None]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                model,
                correction_prompt,
                edit_image_urls,
                out_dir / f"{_slugify(model)}.png",
                aspect_ratio,
            ): model
            for model in edit_models
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    print("\nComparison results")
    print("------------------")
    print(f"Run:                {resolved_run_id}")
    print(f"Composite source:   {composite_name} ({composite_id[:16]})")
    print(f"Correction source:  {critique_name} ({critique_id[:16]})")
    print(f"Output directory:   {out_dir.resolve()}")
    print()
    for model, path, error in sorted(results, key=lambda r: r[0]):
        if error is None:
            print(f"  {model:50s}  ok    -> {path.name}")
        else:
            print(f"  {model:50s}  FAIL  {error}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    compare(
        run_id=args.run_id,
        composite_name=args.composite,
        critique_name=args.critique,
        storage_spec=args.storage,
        edit_models=args.edit_models,
        out_dir=args.out_dir,
        max_workers=args.max_workers,
        aspect_ratio=args.aspect_ratio,
    )


if __name__ == "__main__":
    main()
