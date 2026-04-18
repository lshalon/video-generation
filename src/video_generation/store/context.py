"""``StepContext``: thin facade a step uses to record its outputs to a run.

Steps generally do not need direct access to the :class:`ContentStore` or
:class:`RunStore` — they only need to:

1. Register a byte blob as a step output (returning the content id).
2. Stash arbitrary attributes on the step record (e.g. number of refinements).
3. Materialize an input content id to a local file when an external API
   requires a real filesystem path.

This class wraps the three so step modules stay focused on the work they do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from video_generation.store.content import ContentStore
from video_generation.store.models import STATUS_COMPLETED, STATUS_RUNNING, StepRecord
from video_generation.store.runs import RunStore


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class StepContext:
    """Per-step recording facade passed into each step function."""

    content_store: ContentStore
    run_store: RunStore
    run_id: str
    step_name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    _step: StepRecord | None = None

    def begin(self) -> None:
        """Mark the step as running and persist its input role bindings."""
        self._step = StepRecord(
            name=self.step_name,
            status=STATUS_RUNNING,
            started_at=_utc_now_iso(),
            inputs=dict(self.inputs),
        )
        self.run_store.record_step(self.run_id, self._step)

    def end(self, status: str = STATUS_COMPLETED) -> None:
        """Mark the step finished and persist."""
        step = self._load_step()
        step.status = status
        step.finished_at = _utc_now_iso()
        self.run_store.record_step(self.run_id, step)

    def record(
        self,
        name: str,
        data: bytes,
        original_name: str | None = None,
        mime: str | None = None,
        kind: str | None = None,
    ) -> str:
        """Register ``data`` as content and attach it to this step's outputs.

        Returns the content id. Also tags the content's ``produced_by`` to
        point back at this run+step+output_name.
        """
        ref = self.content_store.register_bytes(
            data=data,
            kind=kind,
            original_name=original_name,
            mime=mime,
            produced_by={"run_id": self.run_id, "step": self.step_name, "name": name},
        )
        self.run_store.attach_output(self.run_id, self.step_name, name, ref.content_id)
        return ref.content_id

    def set_attribute(self, key: str, value: Any) -> None:
        step = self._load_step()
        step.attributes[key] = value
        self.run_store.record_step(self.run_id, step)

    def materialize_input(self, content_id: str, dest_dir: Path) -> Path:
        """Write the bytes for ``content_id`` to ``dest_dir`` and return the path."""
        return self.content_store.materialize(content_id, dest_dir)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_step(self) -> StepRecord:
        run = self.run_store.get(self.run_id)
        step = run.steps.get(self.step_name)
        if step is None:
            self.begin()
            assert self._step is not None
            return self._step
        return step
