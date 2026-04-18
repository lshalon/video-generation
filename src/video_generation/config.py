"""Pipeline configuration and result dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the video generation pipeline."""

    product_dir: Path
    reference_video: Path | None = None
    reference_analysis: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("data/outputs"))
    claude_model: str = "claude-sonnet-4-20250514"
    gemini_model: str = "gemini-3.1-pro-preview"
    edit_model: str = "fal-ai/bytedance/seedream/v4.5/edit"
    video_model: str = "fal-ai/kling-video/v2.6/pro/image-to-video"
    video_duration: str = "5"
    num_frames: int = 20

    def validate(self) -> None:
        """Validate that required inputs are provided."""
        if not self.product_dir.exists():
            raise FileNotFoundError(f"Product directory not found: {self.product_dir}")
        if self.reference_video and not self.reference_video.exists():
            raise FileNotFoundError(f"Reference video not found: {self.reference_video}")
        if self.reference_analysis and not self.reference_analysis.exists():
            raise FileNotFoundError(f"Reference analysis not found: {self.reference_analysis}")
        if not self.reference_video and not self.reference_analysis:
            raise ValueError("Must provide either --reference-video or --reference-analysis")


@dataclass
class AnalysisResult:
    """Output of the analyze_reference step."""

    analysis_text: str
    output_path: Path


@dataclass
class ScriptResult:
    """Output of the write_script step."""

    script_text: str
    script_path: Path


@dataclass
class StartingFrameResult:
    """Output of the generate_starting_frame step."""

    frame_path: Path


@dataclass
class VideoResult:
    """Output of the generate_video step."""

    video_path: Path


@dataclass
class PipelineResult:
    """All outputs from a full pipeline run."""

    analysis_path: Path
    script_path: Path
    starting_frame_path: Path
    video_path: Path

    def summary(self) -> str:
        """Return a human-readable summary of all outputs."""
        lines = [
            "",
            "Pipeline complete! All outputs:",
            "",
            f"  1. Reference analysis : {self.analysis_path}",
            f"  2. Video script       : {self.script_path}",
            f"  3. Starting frame     : {self.starting_frame_path}",
            f"  4. Final video        : {self.video_path}",
            "",
        ]
        return "\n".join(lines)
