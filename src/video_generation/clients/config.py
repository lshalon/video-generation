"""API configuration and client initialization."""

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from anthropic import Anthropic
    from openai import OpenAI


# API key environment variable names
API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",  # Alias
    "kling_access": "KLING_ACCESS_KEY",
    "kling_secret": "KLING_SECRET_KEY",
    "fal": "FAL_KEY",
}

# Project root for finding .env file
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_env(env_file: str | Path | None = None) -> bool:
    """Load environment variables from .env file.

    Args:
        env_file: Path to .env file. If None, searches for .env in project root.

    Returns:
        True if a .env file was found and loaded, False otherwise.
    """
    if env_file is None:
        env_file = PROJECT_ROOT / ".env"

    env_file = Path(env_file)

    if env_file.exists():
        load_dotenv(env_file)
        return True
    return False


def get_api_key(service: str, required: bool = True) -> str | None:
    """Get an API key for a service.

    Args:
        service: Service name (openai, anthropic, claude, kling, fal)
        required: If True, raises an error if the key is not found

    Returns:
        The API key string, or None if not found and not required

    Raises:
        ValueError: If the service is unknown
        EnvironmentError: If required=True and key is not found
    """
    # Ensure env is loaded
    load_env()

    service_lower = service.lower()
    if service_lower not in API_KEYS:
        raise ValueError(
            f"Unknown service: {service}. Valid services: {', '.join(API_KEYS.keys())}"
        )

    env_var = API_KEYS[service_lower]
    key = os.environ.get(env_var)

    if key is None and required:
        raise OSError(
            f"API key not found for {service}. "
            f"Please set {env_var} in your .env file or environment."
        )

    return key


@lru_cache(maxsize=1)
def get_openai_client() -> "OpenAI":
    """Get an OpenAI client instance.

    Returns:
        Configured OpenAI client

    Raises:
        ImportError: If openai package is not installed
        EnvironmentError: If OPENAI_API_KEY is not set
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("OpenAI package not installed. Run: uv add openai") from e

    api_key = get_api_key("openai")
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def get_anthropic_client() -> "Anthropic":
    """Get an Anthropic client instance.

    Returns:
        Configured Anthropic client

    Raises:
        ImportError: If anthropic package is not installed
        EnvironmentError: If ANTHROPIC_API_KEY is not set
    """
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise ImportError("Anthropic package not installed. Run: uv add anthropic") from e

    api_key = get_api_key("anthropic")
    return Anthropic(api_key=api_key)


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are configured.

    Returns:
        Dictionary mapping service names to whether their key is set
    """
    load_env()
    return {
        service: os.environ.get(env_var) is not None
        for service, env_var in API_KEYS.items()
        if service != "claude"  # Skip alias
    }
