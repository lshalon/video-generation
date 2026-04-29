"""Tests for the RunStore."""

import json
from pathlib import Path

import pytest

from video_generation.store import (
    CodeVersion,
    ContentStore,
    LocalStorage,
    RunInputs,
    RunParams,
    RunStore,
    StepContext,
)
from video_generation.store.models import (
    STATUS_COMPLETED,
    STEP_ANALYZE,
    STEP_FRAME,
    StepRecord,
)
from video_generation.store.runs import derive_run_id


@pytest.fixture
def stores(tmp_path: Path) -> tuple[ContentStore, RunStore, LocalStorage]:
    storage = LocalStorage(tmp_path)
    content_store = ContentStore(storage)
    run_store = RunStore(storage, content_store)
    return content_store, run_store, storage


@pytest.fixture
def code_version() -> CodeVersion:
    return CodeVersion(git_sha="abc123", git_dirty=False)


@pytest.fixture
def params() -> RunParams:
    return RunParams(
        claude_model="c",
        gemini_model="g",
        edit_model="e",
        video_model="v",
        video_duration="5",
        max_refinements=3,
    )


@pytest.fixture
def inputs() -> RunInputs:
    return RunInputs(
        product_images=["aa11", "bb22"],
        reference_video="cc33",
        reference_analysis=None,
    )


class TestDeriveRunId:
    def test_deterministic(
        self, inputs: RunInputs, params: RunParams, code_version: CodeVersion
    ) -> None:
        assert derive_run_id(inputs, params, code_version) == derive_run_id(
            inputs, params, code_version
        )

    def test_changes_with_inputs(
        self, inputs: RunInputs, params: RunParams, code_version: CodeVersion
    ) -> None:
        other = RunInputs(product_images=["different"], reference_video="cc33")
        assert derive_run_id(inputs, params, code_version) != derive_run_id(
            other, params, code_version
        )

    def test_changes_with_variant(
        self, inputs: RunInputs, params: RunParams, code_version: CodeVersion
    ) -> None:
        forked = RunParams(**{**params.to_dict(), "variant": "alt"})
        assert derive_run_id(inputs, params, code_version) != derive_run_id(
            inputs, forked, code_version
        )


class TestCreateOrLoad:
    def test_creates_then_resumes(
        self,
        stores: tuple[ContentStore, RunStore, LocalStorage],
        inputs: RunInputs,
        params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        _, run_store, _ = stores
        first = run_store.create_or_load(inputs, params, code_version)
        again = run_store.create_or_load(inputs, params, code_version)
        assert first.run_id == again.run_id
        assert again.started_at == first.started_at

    def test_persists_manifest(
        self,
        stores: tuple[ContentStore, RunStore, LocalStorage],
        inputs: RunInputs,
        params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        _, run_store, storage = stores
        run = run_store.create_or_load(inputs, params, code_version)
        manifest = json.loads(storage.get(f"runs/{run.run_id}/manifest.json"))
        assert manifest["run_id"] == run.run_id
        assert manifest["code_version"]["git_sha"] == "abc123"


class TestRecordStep:
    def test_round_trip(
        self,
        stores: tuple[ContentStore, RunStore, LocalStorage],
        inputs: RunInputs,
        params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        _, run_store, _ = stores
        run = run_store.create_or_load(inputs, params, code_version)
        step = StepRecord(
            name=STEP_ANALYZE, status=STATUS_COMPLETED, outputs={"analysis": "deadbeef"}
        )
        run_store.record_step(run.run_id, step)
        reloaded = run_store.get(run.run_id)
        assert reloaded.steps[STEP_ANALYZE].outputs == {"analysis": "deadbeef"}


class TestAttachOutput:
    def test_writes_convenience_copy(
        self,
        stores: tuple[ContentStore, RunStore, LocalStorage],
        inputs: RunInputs,
        params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        content_store, run_store, storage = stores
        run = run_store.create_or_load(inputs, params, code_version)
        ref = content_store.register_bytes(b"frame", original_name="starting_frame.png")
        run_store.attach_output(run.run_id, STEP_FRAME, "starting_frame", ref.content_id)
        copy_key = f"runs/{run.run_id}/steps/frame/starting_frame.png"
        assert storage.exists(copy_key)
        assert storage.get(copy_key) == b"frame"


class TestStepContext:
    def test_record_attaches_and_returns_id(
        self,
        stores: tuple[ContentStore, RunStore, LocalStorage],
        inputs: RunInputs,
        params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        content_store, run_store, _ = stores
        run = run_store.create_or_load(inputs, params, code_version)
        ctx = StepContext(content_store, run_store, run.run_id, STEP_ANALYZE)
        ctx.begin()
        cid = ctx.record(name="analysis", data=b"# hi", original_name="analysis.md")
        ctx.end()

        reloaded = run_store.get(run.run_id)
        step = reloaded.steps[STEP_ANALYZE]
        assert step.outputs == {"analysis": cid}
        assert step.status == STATUS_COMPLETED

    def test_local_step_output_path(
        self,
        stores: tuple[ContentStore, RunStore, LocalStorage],
        inputs: RunInputs,
        params: RunParams,
        code_version: CodeVersion,
    ) -> None:
        content_store, run_store, storage = stores
        run = run_store.create_or_load(inputs, params, code_version)
        ctx = StepContext(content_store, run_store, run.run_id, STEP_ANALYZE)
        ctx.begin()
        ctx.record(name="analysis", data=b"# hi", original_name="analysis.md")
        ctx.end()

        path = run_store.local_step_output_path(run.run_id, STEP_ANALYZE, "analysis")
        assert path is not None
        assert path == storage.root / "runs" / run.run_id / "steps" / "analyze" / "analysis.md"
        assert path.exists()
