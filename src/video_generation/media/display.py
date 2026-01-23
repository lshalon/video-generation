"""Display utilities for rendering images and videos in Jupyter notebooks."""

import base64
from pathlib import Path

from IPython.display import HTML, Image, Video, display


def display_image(
    source: str | Path | bytes,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Display an image in a Jupyter notebook.

    Args:
        source: Path to image file, URL, or raw bytes
        width: Optional width in pixels
        height: Optional height in pixels
    """
    if isinstance(source, bytes):
        display(Image(data=source, width=width, height=height))
    else:
        path = Path(source) if isinstance(source, str) else source
        if path.exists():
            display(Image(filename=str(path), width=width, height=height))
        else:
            # Assume it's a URL
            display(Image(url=str(source), width=width, height=height))


def display_video(
    source: str | Path | bytes,
    width: int = 640,
    height: int = 480,
    autoplay: bool = False,
    loop: bool = False,
) -> None:
    """Display a video in a Jupyter notebook.

    Args:
        source: Path to video file, URL, or raw bytes
        width: Width in pixels
        height: Height in pixels
        autoplay: Whether to autoplay the video
        loop: Whether to loop the video
    """
    if isinstance(source, bytes):
        # Embed video data directly using base64
        b64_data = base64.b64encode(source).decode("utf-8")
        autoplay_attr = "autoplay" if autoplay else ""
        loop_attr = "loop" if loop else ""
        html = f"""
        <video width="{width}" height="{height}" controls {autoplay_attr} {loop_attr}>
            <source src="data:video/mp4;base64,{b64_data}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        display(HTML(html))
    else:
        path = Path(source) if isinstance(source, str) else source
        if path.exists():
            # Use HTML for local files for better control
            autoplay_attr = "autoplay" if autoplay else ""
            loop_attr = "loop" if loop else ""
            html = f"""
            <video width="{width}" height="{height}" controls {autoplay_attr} {loop_attr}>
                <source src="{path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
            display(HTML(html))
        else:
            # Assume URL, use Video widget
            display(Video(url=str(source), width=width, height=height))


def display_video_file(path: str | Path, width: int = 640) -> None:
    """Display a video file with a simple player.

    This is an alternative method that reads the file and embeds it.

    Args:
        path: Path to the video file
        width: Width of the player in pixels
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    with open(path, "rb") as f:
        video_bytes = f.read()

    display_video(video_bytes, width=width)
