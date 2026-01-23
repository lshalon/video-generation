"""Prompt loading and management utilities."""

from pathlib import Path
from string import Template

# Default prompts directory relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"


def load_prompt(
    name: str,
    category: str = "examples",
    prompts_dir: Path | None = None,
    **variables: str,
) -> str:
    """Load a prompt from a text file.

    Args:
        name: Name of the prompt file (with or without .txt extension)
        category: Subdirectory within prompts (e.g., "examples", "system")
        prompts_dir: Custom prompts directory
        **variables: Variables to substitute in the prompt using $variable syntax

    Returns:
        The prompt text with any variables substituted

    Example:
        >>> load_prompt("image_generation", category="examples")
        'A cinematic wide shot...'

        >>> load_prompt("template", subject="mountain", style="cinematic")
        'Create a cinematic image of mountain...'
    """
    if prompts_dir is None:
        prompts_dir = PROMPTS_DIR

    prompts_dir = Path(prompts_dir)

    # Add .txt extension if not present
    if not name.endswith(".txt"):
        name = f"{name}.txt"

    prompt_path = prompts_dir / category / name

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    text = prompt_path.read_text().strip()

    # Substitute variables if provided
    if variables:
        template = Template(text)
        text = template.safe_substitute(**variables)

    return text


def load_system_prompt(name: str, prompts_dir: Path | None = None, **variables: str) -> str:
    """Load a system prompt (shortcut for category='system').

    Args:
        name: Name of the prompt file
        prompts_dir: Custom prompts directory
        **variables: Variables to substitute

    Returns:
        The system prompt text
    """
    return load_prompt(name, category="system", prompts_dir=prompts_dir, **variables)


def list_prompts(
    category: str | None = None,
    prompts_dir: Path | None = None,
) -> dict[str, list[str]]:
    """List all available prompts.

    Args:
        category: Optional category to filter by
        prompts_dir: Custom prompts directory

    Returns:
        Dictionary mapping categories to lists of prompt names
    """
    if prompts_dir is None:
        prompts_dir = PROMPTS_DIR

    prompts_dir = Path(prompts_dir)

    if not prompts_dir.exists():
        return {}

    result: dict[str, list[str]] = {}

    if category:
        cat_dir = prompts_dir / category
        if cat_dir.exists():
            prompts = [f.stem for f in cat_dir.glob("*.txt")]
            if prompts:
                result[category] = sorted(prompts)
    else:
        for cat_dir in prompts_dir.iterdir():
            if cat_dir.is_dir() and not cat_dir.name.startswith("."):
                prompts = [f.stem for f in cat_dir.glob("*.txt")]
                if prompts:
                    result[cat_dir.name] = sorted(prompts)

    return result


def save_prompt(
    name: str,
    content: str,
    category: str = "examples",
    prompts_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Save a prompt to a text file.

    Args:
        name: Name for the prompt file (with or without .txt extension)
        content: The prompt text to save
        category: Subdirectory within prompts
        prompts_dir: Custom prompts directory
        overwrite: Whether to overwrite existing files

    Returns:
        Path to the saved prompt file

    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    if prompts_dir is None:
        prompts_dir = PROMPTS_DIR

    prompts_dir = Path(prompts_dir)

    # Add .txt extension if not present
    if not name.endswith(".txt"):
        name = f"{name}.txt"

    cat_dir = prompts_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = cat_dir / name

    if prompt_path.exists() and not overwrite:
        raise FileExistsError(
            f"Prompt already exists: {prompt_path}. Use overwrite=True to replace."
        )

    prompt_path.write_text(content.strip() + "\n")
    return prompt_path
