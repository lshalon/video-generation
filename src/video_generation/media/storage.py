"""Storage utilities for saving images and videos with consistent naming."""

import hashlib
from datetime import datetime
from pathlib import Path

# Default output directories relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "outputs" / "images"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "outputs" / "videos"


def get_timestamp() -> str:
    """Get a timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_short_hash(data: bytes, length: int = 8) -> str:
    """Get a short hash of data for unique identification."""
    return hashlib.sha256(data).hexdigest()[:length]


def get_output_path(
    filename: str | None = None,
    prefix: str = "",
    suffix: str = "",
    extension: str = "png",
    media_type: str = "image",
    output_dir: Path | None = None,
) -> Path:
    """Generate a consistent output path for saved media.

    Args:
        filename: Optional custom filename (without extension)
        prefix: Optional prefix to add to generated filename
        suffix: Optional suffix to add to generated filename
        extension: File extension (default: png for images)
        media_type: Either "image" or "video"
        output_dir: Custom output directory (uses defaults if None)

    Returns:
        Path object for the output file
    """
    if output_dir is None:
        output_dir = DEFAULT_IMAGE_DIR if media_type == "image" else DEFAULT_VIDEO_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        name = filename
    else:
        timestamp = get_timestamp()
        parts = [p for p in [prefix, timestamp, suffix] if p]
        name = "_".join(parts)

    return output_dir / f"{name}.{extension}"


def save_image(
    data: bytes,
    filename: str | None = None,
    prefix: str = "",
    suffix: str = "",
    extension: str = "png",
    output_dir: Path | None = None,
    include_hash: bool = True,
) -> Path:
    """Save image data to a file with consistent naming.

    Args:
        data: Raw image bytes
        filename: Optional custom filename (without extension)
        prefix: Optional prefix for generated filename
        suffix: Optional suffix for generated filename
        extension: File extension (default: png)
        output_dir: Custom output directory
        include_hash: Include content hash in filename for uniqueness

    Returns:
        Path to the saved file
    """
    if include_hash and not filename:
        hash_suffix = get_short_hash(data)
        suffix = f"{suffix}_{hash_suffix}" if suffix else hash_suffix

    path = get_output_path(
        filename=filename,
        prefix=prefix,
        suffix=suffix,
        extension=extension,
        media_type="image",
        output_dir=output_dir,
    )

    path.write_bytes(data)
    return path


def save_video(
    data: bytes,
    filename: str | None = None,
    prefix: str = "",
    suffix: str = "",
    extension: str = "mp4",
    output_dir: Path | None = None,
    include_hash: bool = True,
) -> Path:
    """Save video data to a file with consistent naming.

    Args:
        data: Raw video bytes
        filename: Optional custom filename (without extension)
        prefix: Optional prefix for generated filename
        suffix: Optional suffix for generated filename
        extension: File extension (default: mp4)
        output_dir: Custom output directory
        include_hash: Include content hash in filename for uniqueness

    Returns:
        Path to the saved file
    """
    if include_hash and not filename:
        hash_suffix = get_short_hash(data)
        suffix = f"{suffix}_{hash_suffix}" if suffix else hash_suffix

    path = get_output_path(
        filename=filename,
        prefix=prefix,
        suffix=suffix,
        extension=extension,
        media_type="video",
        output_dir=output_dir,
    )

    path.write_bytes(data)
    return path


def list_outputs(media_type: str = "image", output_dir: Path | None = None) -> list[Path]:
    """List all output files of a given type.

    Args:
        media_type: Either "image" or "video"
        output_dir: Custom output directory

    Returns:
        List of paths to output files, sorted by modification time (newest first)
    """
    if output_dir is None:
        output_dir = DEFAULT_IMAGE_DIR if media_type == "image" else DEFAULT_VIDEO_DIR

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    files = list(output_dir.iterdir())
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
