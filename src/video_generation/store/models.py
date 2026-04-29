"""Dataclasses for the content store and run records.

The structures here are the on-disk schema for ``library/<aa>/<sha>.meta.json``
and ``runs/<run_id>/manifest.json``. They are intentionally JSON-friendly:
``to_dict``/``from_dict`` round-trip through ``json.dumps``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Status strings are kept as plain literals (rather than an Enum) so the
# manifest JSON stays simple and human-editable.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Step names used across the pipeline. Keep in sync with the CLI's VALID_STEPS.
STEP_ANALYZE = "analyze"
STEP_SCRIPT = "script"
STEP_FRAME = "frame"
STEP_VIDEO = "video"


@dataclass(frozen=True)
class ContentMeta:
    """Metadata sidecar stored next to each content blob."""

    sha256: str
    size: int
    mime: str | None
    kind: str
    original_name: str | None
    registered_at: str
    produced_by: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentMeta:
        return cls(
            sha256=data["sha256"],
            size=int(data["size"]),
            mime=data.get("mime"),
            kind=data["kind"],
            original_name=data.get("original_name"),
            registered_at=data["registered_at"],
            produced_by=data.get("produced_by"),
        )


@dataclass(frozen=True)
class ContentRef:
    """A handle to a specific content blob in the store."""

    content_id: str
    meta: ContentMeta

    @property
    def short_id(self) -> str:
        return self.content_id[:16]


@dataclass(frozen=True)
class CodeVersion:
    """Snapshot of the code that produced a run."""

    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, Any]:
        return {"git_sha": self.git_sha, "git_dirty": self.git_dirty}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeVersion:
        return cls(git_sha=data["git_sha"], git_dirty=bool(data["git_dirty"]))


@dataclass(frozen=True)
class RunInputs:
    """Content IDs supplied as inputs to a run, organized by role."""

    product_images: list[str] = field(default_factory=list)
    reference_video: str | None = None
    reference_analysis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_images": list(self.product_images),
            "reference_video": self.reference_video,
            "reference_analysis": self.reference_analysis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunInputs:
        return cls(
            product_images=list(data.get("product_images", [])),
            reference_video=data.get("reference_video"),
            reference_analysis=data.get("reference_analysis"),
        )


@dataclass(frozen=True)
class RunParams:
    """Run-level parameters that influence outputs (model identifiers, etc.).

    ``variant`` is a free-form fork tag: same inputs + code + params but
    different ``variant`` give different run ids.
    """

    claude_model: str
    gemini_model: str
    edit_model: str
    video_model: str
    video_duration: str
    max_refinements: int
    variant: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunParams:
        return cls(
            claude_model=data["claude_model"],
            gemini_model=data["gemini_model"],
            edit_model=data["edit_model"],
            video_model=data["video_model"],
            video_duration=data["video_duration"],
            max_refinements=int(data["max_refinements"]),
            variant=data.get("variant", ""),
        )


@dataclass
class StepRecord:
    """Record of a single step's execution within a run."""

    name: str
    status: str = STATUS_PENDING
    started_at: str | None = None
    finished_at: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        return cls(
            name=data["name"],
            status=data.get("status", STATUS_PENDING),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            inputs=dict(data.get("inputs", {})),
            outputs=dict(data.get("outputs", {})),
            attributes=dict(data.get("attributes", {})),
        )

    @property
    def is_completed(self) -> bool:
        return self.status == STATUS_COMPLETED


@dataclass
class RunRecord:
    """Top-level manifest for a single run."""

    run_id: str
    status: str
    started_at: str
    finished_at: str | None
    code_version: CodeVersion
    params: RunParams
    inputs: RunInputs
    steps: dict[str, StepRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "code_version": self.code_version.to_dict(),
            "params": self.params.to_dict(),
            "inputs": self.inputs.to_dict(),
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        steps_data = data.get("steps", {})
        return cls(
            run_id=data["run_id"],
            status=data.get("status", STATUS_PENDING),
            started_at=data["started_at"],
            finished_at=data.get("finished_at"),
            code_version=CodeVersion.from_dict(data["code_version"]),
            params=RunParams.from_dict(data["params"]),
            inputs=RunInputs.from_dict(data["inputs"]),
            steps={name: StepRecord.from_dict(payload) for name, payload in steps_data.items()},
        )

    def get_step(self, name: str) -> StepRecord | None:
        return self.steps.get(name)
