"""Persistent record of pipeline runs.

A :class:`RunStore` writes one ``runs/<run_id>/manifest.json`` per run and
tracks per-step outputs (which are themselves :class:`ContentRef` registered
in the :class:`ContentStore`). Optionally lays down ``runs/<run_id>/steps/...``
convenience copies that mirror the most recent step outputs by name.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from video_generation.store.content import ContentStore
from video_generation.store.models import (
    STATUS_COMPLETED,
    STATUS_RUNNING,
    CodeVersion,
    RunInputs,
    RunParams,
    RunRecord,
    StepRecord,
)
from video_generation.store.storage import LocalStorage, Storage

logger = logging.getLogger(__name__)

RUNS_PREFIX = "runs"
RUN_ID_LEN = 16


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest_key(run_id: str) -> str:
    return f"{RUNS_PREFIX}/{run_id}/manifest.json"


def _step_copy_key(run_id: str, step_name: str, output_name: str, suffix: str) -> str:
    return f"{RUNS_PREFIX}/{run_id}/steps/{step_name}/{output_name}{suffix}"


def derive_run_id(
    inputs: RunInputs,
    params: RunParams,
    code_version: CodeVersion,
) -> str:
    """Deterministic run id derived from inputs + params + code_version.

    Same triple -> same run_id (so a re-run is idempotent and resumable). To
    fork with the same triple, set ``params.variant`` to a unique tag.
    """
    payload = {
        "inputs": inputs.to_dict(),
        "params": params.to_dict(),
        "code_version": code_version.to_dict(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:RUN_ID_LEN]


class RunStore:
    """High-level interface for creating, loading, and updating run records."""

    def __init__(self, storage: Storage, content_store: ContentStore) -> None:
        self.storage = storage
        self.content_store = content_store

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_or_load(
        self,
        inputs: RunInputs,
        params: RunParams,
        code_version: CodeVersion,
    ) -> RunRecord:
        """Return an existing run with this triple, or create a new one."""
        run_id = derive_run_id(inputs, params, code_version)
        if self.storage.exists(_manifest_key(run_id)):
            logger.info("Resuming existing run: %s", run_id)
            return self.get(run_id)

        run = RunRecord(
            run_id=run_id,
            status=STATUS_RUNNING,
            started_at=_utc_now_iso(),
            finished_at=None,
            code_version=code_version,
            params=params,
            inputs=inputs,
            steps={},
        )
        self._persist(run)
        logger.info("Created new run: %s", run_id)
        return run

    def get(self, run_id: str) -> RunRecord:
        key = _manifest_key(run_id)
        if not self.storage.exists(key):
            raise KeyError(f"Run not found: {run_id}")
        return RunRecord.from_dict(json.loads(self.storage.get(key)))

    def mark_finished(self, run_id: str, status: str = STATUS_COMPLETED) -> RunRecord:
        run = self.get(run_id)
        run.status = status
        run.finished_at = _utc_now_iso()
        self._persist(run)
        return run

    # ------------------------------------------------------------------
    # Step recording
    # ------------------------------------------------------------------

    def record_step(self, run_id: str, step: StepRecord) -> RunRecord:
        """Replace (or insert) ``step`` in the run manifest."""
        run = self.get(run_id)
        run.steps[step.name] = step
        self._persist(run)
        return run

    def attach_output(
        self,
        run_id: str,
        step_name: str,
        output_name: str,
        content_id: str,
        write_convenience_copy: bool = True,
    ) -> StepRecord:
        """Record a single step output in the manifest.

        When ``write_convenience_copy`` is true, the bytes are also written
        under ``runs/<run_id>/steps/<step>/<output_name><ext>`` so a human
        can browse a run directory without resolving content ids.
        """
        run = self.get(run_id)
        step = run.steps.get(step_name)
        if step is None:
            step = StepRecord(name=step_name, status=STATUS_RUNNING, started_at=_utc_now_iso())
            run.steps[step_name] = step
        step.outputs[output_name] = content_id

        if write_convenience_copy:
            meta = self.content_store.get_meta(content_id)
            suffix = ""
            if meta.original_name:
                suffix = Path(meta.original_name).suffix.lower()
            elif meta.kind == "json":
                suffix = ".json"
            elif meta.kind == "text":
                suffix = ".txt"
            self.storage.put(
                _step_copy_key(run_id, step_name, output_name, suffix),
                self.content_store.get_bytes(content_id),
                mime=meta.mime,
            )

        self._persist(run)
        return step

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def local_step_output_path(
        self,
        run_id: str,
        step_name: str,
        output_name: str,
    ) -> Path | None:
        """Return the on-disk path of a step output's convenience copy.

        Returns ``None`` if the underlying storage is not :class:`LocalStorage`,
        or if no output by that name has been recorded.
        """
        if not isinstance(self.storage, LocalStorage):
            return None
        run = self.get(run_id)
        step = run.steps.get(step_name)
        if step is None:
            return None
        content_id = step.outputs.get(output_name)
        if content_id is None:
            return None
        meta = self.content_store.get_meta(content_id)
        suffix = ""
        if meta.original_name:
            suffix = Path(meta.original_name).suffix.lower()
        elif meta.kind == "json":
            suffix = ".json"
        elif meta.kind == "text":
            suffix = ".txt"
        return (
            self.storage.root
            / RUNS_PREFIX
            / run_id
            / "steps"
            / step_name
            / f"{output_name}{suffix}"
        )

    def _persist(self, run: RunRecord) -> None:
        self.storage.put(
            _manifest_key(run.run_id),
            json.dumps(run.to_dict(), indent=2).encode("utf-8"),
            mime="application/json",
        )
