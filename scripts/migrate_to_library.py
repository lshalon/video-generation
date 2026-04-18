"""Register existing data/inputs/jewelry/ files into the content store.

After registering, emits two human-browsable "collection" JSONs:

- ``library/collections/products/jewelry-earrings.json``
- ``library/collections/references/yurman-full-shot.json``

Run with::

    uv run python -m scripts.migrate_to_library

or::

    uv run python scripts/migrate_to_library.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from video_generation.steps.write_script import discover_product_images
from video_generation.store import ContentStore, make_storage

logger = logging.getLogger(__name__)

DEFAULT_PRODUCT_DIR = Path("data/inputs/jewelry/product")
DEFAULT_REFERENCE_VIDEO = Path("data/inputs/jewelry/reference/yurman-full-shot.mp4")
DEFAULT_REFERENCE_ANALYSIS = Path("data/inputs/jewelry/reference/yurman-full-shot-analysis.md")
PRODUCT_COLLECTION_KEY = "library/collections/products/jewelry-earrings.json"
REFERENCE_COLLECTION_KEY = "library/collections/references/yurman-full-shot.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage",
        type=str,
        default="local:./data",
        help="Storage spec to register into (default: local:./data).",
    )
    parser.add_argument(
        "--product-dir",
        type=Path,
        default=DEFAULT_PRODUCT_DIR,
        help=f"Product image directory (default: {DEFAULT_PRODUCT_DIR}).",
    )
    parser.add_argument(
        "--reference-video",
        type=Path,
        default=DEFAULT_REFERENCE_VIDEO,
        help=f"Reference video file (default: {DEFAULT_REFERENCE_VIDEO}).",
    )
    parser.add_argument(
        "--reference-analysis",
        type=Path,
        default=DEFAULT_REFERENCE_ANALYSIS,
        help=f"Reference analysis markdown (default: {DEFAULT_REFERENCE_ANALYSIS}).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _register_products(content_store: ContentStore, product_dir: Path) -> dict[str, str]:
    if not product_dir.exists():
        logger.warning("Product directory missing: %s", product_dir)
        return {}
    items: dict[str, str] = {}
    for path in discover_product_images(product_dir):
        ref = content_store.register_path(path, kind="image")
        items[path.name] = ref.content_id
        logger.info("Registered product %s -> %s", path.name, ref.short_id)
    return items


def _register_one(content_store: ContentStore, path: Path, kind: str) -> str | None:
    if not path.exists():
        logger.warning("Missing file (skipped): %s", path)
        return None
    ref = content_store.register_path(path, kind=kind)
    logger.info("Registered %s -> %s", path.name, ref.short_id)
    return ref.content_id


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    storage = make_storage(args.storage)
    content_store = ContentStore(storage)

    products = _register_products(content_store, args.product_dir)
    reference_video_id = _register_one(content_store, args.reference_video, kind="video")
    reference_analysis_id = _register_one(content_store, args.reference_analysis, kind="text")

    if products:
        product_collection: dict[str, Any] = {
            "name": "Jewelry earrings (Yurman demo)",
            "kind": "product",
            "items": products,
        }
        storage.put(
            PRODUCT_COLLECTION_KEY,
            json.dumps(product_collection, indent=2).encode("utf-8"),
            mime="application/json",
        )
        logger.info("Wrote collection: %s", PRODUCT_COLLECTION_KEY)

    if reference_video_id or reference_analysis_id:
        reference_collection: dict[str, Any] = {
            "name": "Yurman full-shot reference",
            "kind": "reference",
            "video": reference_video_id,
            "analysis": reference_analysis_id,
        }
        storage.put(
            REFERENCE_COLLECTION_KEY,
            json.dumps(reference_collection, indent=2).encode("utf-8"),
            mime="application/json",
        )
        logger.info("Wrote collection: %s", REFERENCE_COLLECTION_KEY)

    print("Migration complete.")
    print(f"  Products registered : {len(products)}")
    if reference_video_id:
        print(f"  Reference video     : {reference_video_id[:16]}")
    if reference_analysis_id:
        print(f"  Reference analysis  : {reference_analysis_id[:16]}")


if __name__ == "__main__":
    main()
