"""Prompt management utilities."""

from video_generation.prompts.loader import (
    PROMPTS_DIR,
    list_prompts,
    load_prompt,
    load_system_prompt,
    save_prompt,
)

__all__ = [
    "load_prompt",
    "load_system_prompt",
    "list_prompts",
    "save_prompt",
    "PROMPTS_DIR",
]
