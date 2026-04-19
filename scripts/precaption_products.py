"""Pre-generate Gemini captions for every product image in a directory.

Captions are stored in the content store next to each blob
(``library/<aa>/<sha>.caption.json``) so subsequent pipeline runs reuse them
verbatim without spending any LLM tokens.

Run with::

    uv run python -m scripts.precaption_products --product-dir data/inputs/jewelry/product

or::

    uv run python scripts/precaption_products.py --product-dir data/inputs/jewelry/product

The script is idempotent: photos that already have a caption are skipped
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from video_generation.steps.generate_starting_frame import (
    DEFAULT_GEMINI_MODEL,
    ensure_product_captions,
)
from video_generation.steps.write_script import discover_product_images
from video_generation.store import ContentStore, make_storage

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product-dir",
        type=Path,
        required=True,
        help="Directory of product images to caption.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="local:./data",
        help="Storage spec for the content store (default: local:./data).",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model identifier (default: {DEFAULT_GEMINI_MODEL}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-caption every image even if a caption already exists.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def precaption(
    product_dir: Path,
    storage_spec: str,
    gemini_model: str,
    force: bool,
) -> None:
    if not product_dir.is_dir():
        raise FileNotFoundError(f"Product directory not found: {product_dir}")

    storage = make_storage(storage_spec)
    content_store = ContentStore(storage)

    image_paths = discover_product_images(product_dir)
    if not image_paths:
        logger.warning("No product images found under %s", product_dir)
        return

    refs = [content_store.register_path(p, kind="image") for p in image_paths]
    content_ids = [ref.content_id for ref in refs]

    if force:
        cleared = 0
        for cid in content_ids:
            if content_store.delete_caption(cid):
                cleared += 1
        logger.info("--force: cleared %d existing caption(s)", cleared)

    captions, generated, reused = ensure_product_captions(content_ids, content_store, gemini_model)

    print(
        f"Captioned {len(captions)} unique image(s) "
        f"({generated} freshly generated, {reused} reused from library):"
    )
    for cid, path in zip(content_ids, image_paths, strict=True):
        print(f"  [{cid[:16]}] {path.name}")
        print(f"      {captions[cid]}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    precaption(
        product_dir=args.product_dir,
        storage_spec=args.storage,
        gemini_model=args.gemini_model,
        force=args.force,
    )


if __name__ == "__main__":
    main()
