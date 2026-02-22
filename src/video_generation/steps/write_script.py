"""Step 2: Generate a video script from reference analysis and product images."""

import base64
import logging
from pathlib import Path
from string import Template

import cv2

from video_generation.clients import get_anthropic_client
from video_generation.config import ScriptResult
from video_generation.prompts import load_prompt, load_system_prompt

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png")


def load_product_images(product_dir: Path) -> list[Path]:
    """Find all product images in a directory.

    Args:
        product_dir: Directory containing product images.

    Returns:
        Sorted list of image file paths.
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


def load_image_as_base64(path: Path, max_size: int = 800) -> tuple[str, str]:
    """Load an image and convert to base64-encoded JPEG.

    Args:
        path: Path to the image file.
        max_size: Maximum dimension (width or height).

    Returns:
        Tuple of (base64_data, media_type).
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.standard_b64encode(buffer).decode("utf-8")
    return b64, "image/jpeg"


def write_script(
    reference_analysis_path: Path,
    product_dir: Path,
    *,
    claude_model: str = "claude-sonnet-4-20250514",
    output_dir: Path | None = None,
) -> ScriptResult:
    """Generate a video script that emulates a reference style for a product.

    Args:
        reference_analysis_path: Path to the reference analysis markdown.
        product_dir: Directory containing product images.
        claude_model: Claude model identifier.
        output_dir: Directory for the script output.

    Returns:
        ScriptResult with script text and output path.
    """
    logger.info("Writing script from analysis: %s", reference_analysis_path)

    reference_analysis = reference_analysis_path.read_text()
    product_images = load_product_images(product_dir)
    logger.info("Found %d product images", len(product_images))

    product_description = "Product: Earrings\n\nAvailable product shots:\n"
    for img in product_images:
        name = img.stem.replace("_", " ").title()
        product_description += f"- {name}\n"

    system_prompt = load_system_prompt("script_writer")
    user_prompt_template = load_prompt("emulate_reference_script", category="examples")
    user_prompt = Template(user_prompt_template).substitute(
        reference_analysis=reference_analysis,
        product_description=product_description,
    )

    image_content = []
    for path in product_images:
        b64, media_type = load_image_as_base64(path)
        image_content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            }
        )

    content = image_content + [
        {
            "type": "text",
            "text": f"Here are the product images I have available.\n\n{user_prompt}",
        }
    ]

    client = get_anthropic_client()
    logger.info("Generating script with Claude...")

    response = client.messages.create(
        model=claude_model,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],  # type: ignore[typeddict-item]
    )

    script: str = response.content[0].text  # type: ignore[union-attr]
    logger.info(
        "Script complete (input: %d, output: %d tokens)",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    if output_dir is None:
        output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = output_dir / "script.md"
    script_path.write_text(script)
    logger.info("Script saved to: %s", script_path)

    return ScriptResult(script_text=script, script_path=script_path)
