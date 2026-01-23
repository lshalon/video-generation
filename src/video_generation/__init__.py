"""Video generation monorepo."""

__version__ = "0.1.0"

# Re-export commonly used utilities
from video_generation.clients import get_api_key, load_env
from video_generation.media import display_image, display_video, save_image, save_video
from video_generation.prompts import list_prompts, load_prompt, load_system_prompt, save_prompt

__all__ = [
    "__version__",
    # Media
    "display_image",
    "display_video",
    "save_image",
    "save_video",
    # Prompts
    "load_prompt",
    "load_system_prompt",
    "list_prompts",
    "save_prompt",
    # Config
    "load_env",
    "get_api_key",
]
