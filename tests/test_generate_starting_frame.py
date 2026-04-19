"""Tests for the generate_starting_frame step."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from video_generation.steps.generate_starting_frame import (
    OUTPUT_FINAL,
    STEP_NAME,
    CompositesCritique,
    generate_starting_frame,
)
from video_generation.store import (
    CodeVersion,
    ContentStore,
    RunInputs,
    RunParams,
    RunStore,
    StepContext,
)


def _fake_image_bytes() -> bytes:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return bytes(buf.tobytes())


def _gemini_critique_response(acceptable: bool, issues: list[str], correction: str) -> MagicMock:
    critique = CompositesCritique(
        acceptable=acceptable, issues=issues, correction_prompt=correction
    )
    resp = MagicMock()
    resp.text = json.dumps(critique.model_dump())
    return resp


_PRODUCT_CAPTIONS = (
    "ON A REAL MODEL'S EAR \u2014 left lobe, three-quarter angle, use as size ground truth.",
    "FRONT VIEW \u2014 head-on macro of the earring, shows shape and stones.",
    "SIDE VIEW \u2014 profile of the earring, shows post depth and lobe seating.",
)


def _caption_responses() -> list[MagicMock]:
    """One Gemini caption response per product image in the test fixture (3 images)."""
    return [MagicMock(text=text) for text in _PRODUCT_CAPTIONS]


def _setup(
    content_store: ContentStore,
    run_store: RunStore,
    product_dir: Path,
    script_text: str,
    params: RunParams,
    code_version: CodeVersion,
) -> tuple[StepContext, str, list[str]]:
    script_ref = content_store.register_bytes(
        script_text.encode("utf-8"), original_name="script.md", kind="text"
    )
    product_refs = [
        content_store.register_path(p, kind="image")
        for p in sorted(product_dir.iterdir())
        if p.suffix.lower() in (".png", ".jpg", ".webp")
    ]
    product_ids = [r.content_id for r in product_refs]

    inputs = RunInputs(product_images=product_ids, reference_analysis=script_ref.content_id)
    run = run_store.create_or_load(inputs=inputs, params=params, code_version=code_version)
    ctx = StepContext(
        content_store,
        run_store,
        run.run_id,
        STEP_NAME,
        inputs={"script": script_ref.content_id, "product_images": product_ids},
    )
    return ctx, script_ref.content_id, product_ids


class TestGenerateStartingFrame:
    @patch("video_generation.steps.generate_starting_frame.httpx")
    @patch("video_generation.steps.generate_starting_frame.fal_client")
    @patch("video_generation.steps.generate_starting_frame.get_gemini_client")
    def test_accepted_on_first_try(
        self,
        mock_get_gemini: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        product_dir: Path,
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
        script_text: str,
    ) -> None:
        ctx, script_id, product_ids = _setup(
            content_store, run_store, product_dir, script_text, default_params, code_version
        )

        mock_gemini = MagicMock()
        mock_get_gemini.return_value = mock_gemini
        mock_gemini.models.generate_content.side_effect = [
            MagicMock(text="Tight close-up, unadorned earlobe."),
            *_caption_responses(),
            MagicMock(text="Place the product on the ear."),
            _gemini_critique_response(True, [], ""),
            MagicMock(text="Outpaint to a 3:4 portrait, preserving the ear region exactly."),
        ]

        fake_bytes = _fake_image_bytes()
        mock_fal.subscribe.side_effect = [
            {"images": [{"url": "https://fake/closeup.png"}]},
            {"images": [{"url": "https://fake/composite.png"}]},
            {"images": [{"url": "https://fake/full_frame.png"}]},
        ]
        mock_fal.upload.return_value = "https://fake/uploaded.png"

        mock_resp = MagicMock()
        mock_resp.content = fake_bytes
        mock_httpx.get.return_value = mock_resp

        result = generate_starting_frame(
            script_id,
            product_ids,
            ctx=ctx,
            max_refinements=3,
        )

        assert result.content_id is not None
        assert result.frame_path.exists()
        assert result.frame_path.suffix == ".png"

        run = run_store.get(ctx.run_id)
        outputs = run.steps[STEP_NAME].outputs
        assert outputs[OUTPUT_FINAL] == result.content_id
        assert "closeup_scene_prompt" in outputs
        assert "closeup_scene" in outputs
        assert "composite_prompt" in outputs
        assert "product_captions" in outputs
        assert "closeup_composite_v0" in outputs
        assert "critique_v1" in outputs
        assert "closeup_composite_final" in outputs
        assert "expand_prompt" in outputs

        captions_payload = json.loads(
            content_store.get_bytes(outputs["product_captions"]).decode("utf-8")
        )
        assert {entry["caption"] for entry in captions_payload} == set(_PRODUCT_CAPTIONS)
        assert {entry["content_id"] for entry in captions_payload} == set(product_ids)

        attrs = run.steps[STEP_NAME].attributes
        assert attrs["closeup_accepted"] is True
        assert attrs["closeup_refinements_done"] == 0
        assert attrs["outpaint_done"] is True
        assert attrs["used_reference_video"] is False
        assert attrs["used_reference_analysis"] is False
        assert attrs["captions_generated"] == 3
        assert attrs["captions_reused"] == 0

        persisted_captions = {content_store.get_caption(cid) for cid in product_ids}
        assert persisted_captions == set(_PRODUCT_CAPTIONS)
        for cid in product_ids:
            record = content_store.get_caption_record(cid)
            assert record is not None
            assert record["text"] in _PRODUCT_CAPTIONS
            assert "model" in record
            assert "generated_at" in record

        # 1 closeup-scene + 3 captions + 1 composite + 1 critique + 1 expand = 7
        assert mock_gemini.models.generate_content.call_count == 7
        # 1 closeup t2i + 1 composite edit + 1 outpaint edit = 3
        assert mock_fal.subscribe.call_count == 3

    @patch("video_generation.steps.generate_starting_frame.httpx")
    @patch("video_generation.steps.generate_starting_frame.fal_client")
    @patch("video_generation.steps.generate_starting_frame.get_gemini_client")
    def test_refines_once_then_accepted(
        self,
        mock_get_gemini: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        product_dir: Path,
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
        script_text: str,
    ) -> None:
        ctx, script_id, product_ids = _setup(
            content_store, run_store, product_dir, script_text, default_params, code_version
        )

        mock_gemini = MagicMock()
        mock_get_gemini.return_value = mock_gemini
        mock_gemini.models.generate_content.side_effect = [
            MagicMock(text="Tight close-up scene."),
            *_caption_responses(),
            MagicMock(text="Place product on ear."),
            _gemini_critique_response(False, ["Product too large"], "Make 50% smaller."),
            _gemini_critique_response(True, [], ""),
            MagicMock(text="Outpaint to portrait."),
        ]

        fake_bytes = _fake_image_bytes()
        mock_fal.subscribe.side_effect = [
            {"images": [{"url": "https://fake/closeup.png"}]},
            {"images": [{"url": "https://fake/v0.png"}]},
            {"images": [{"url": "https://fake/v1.png"}]},
            {"images": [{"url": "https://fake/full_frame.png"}]},
        ]
        mock_fal.upload.return_value = "https://fake/uploaded.png"

        mock_resp = MagicMock()
        mock_resp.content = fake_bytes
        mock_httpx.get.return_value = mock_resp

        result = generate_starting_frame(
            script_id,
            product_ids,
            ctx=ctx,
            max_refinements=3,
        )

        assert result.content_id is not None
        run = run_store.get(ctx.run_id)
        outputs = run.steps[STEP_NAME].outputs
        assert "critique_v1" in outputs
        assert "critique_v2" in outputs
        assert "closeup_composite_v1" in outputs
        assert "closeup_composite_final" in outputs
        assert "expand_prompt" in outputs
        attrs = run.steps[STEP_NAME].attributes
        assert attrs["closeup_accepted"] is True
        assert attrs["closeup_refinements_done"] == 1
        assert attrs["outpaint_done"] is True

    @patch("video_generation.steps.generate_starting_frame.httpx")
    @patch("video_generation.steps.generate_starting_frame.fal_client")
    @patch("video_generation.steps.generate_starting_frame.get_gemini_client")
    def test_stops_at_max_refinements(
        self,
        mock_get_gemini: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        product_dir: Path,
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
        script_text: str,
    ) -> None:
        ctx, script_id, product_ids = _setup(
            content_store, run_store, product_dir, script_text, default_params, code_version
        )

        mock_gemini = MagicMock()
        mock_get_gemini.return_value = mock_gemini
        mock_gemini.models.generate_content.side_effect = [
            MagicMock(text="Tight close-up scene."),
            *_caption_responses(),
            MagicMock(text="Place product."),
            _gemini_critique_response(False, ["Still too large"], "Make smaller."),
            _gemini_critique_response(False, ["Still too large"], "Make smaller."),
            MagicMock(text="Outpaint to portrait."),
        ]

        fake_bytes = _fake_image_bytes()
        mock_fal.subscribe.side_effect = [
            {"images": [{"url": "https://fake/closeup.png"}]},
            {"images": [{"url": "https://fake/v0.png"}]},
            {"images": [{"url": "https://fake/v1.png"}]},
            {"images": [{"url": "https://fake/v2.png"}]},
            {"images": [{"url": "https://fake/full_frame.png"}]},
        ]
        mock_fal.upload.return_value = "https://fake/uploaded.png"

        mock_resp = MagicMock()
        mock_resp.content = fake_bytes
        mock_httpx.get.return_value = mock_resp

        result = generate_starting_frame(
            script_id,
            product_ids,
            ctx=ctx,
            max_refinements=2,
        )

        assert result.content_id is not None
        run = run_store.get(ctx.run_id)
        attrs = run.steps[STEP_NAME].attributes
        assert attrs["closeup_accepted"] is False
        assert attrs["closeup_refinements_done"] == 2
        assert attrs["outpaint_done"] is True

    @patch("video_generation.steps.generate_starting_frame.httpx")
    @patch("video_generation.steps.generate_starting_frame.fal_client")
    @patch("video_generation.steps.generate_starting_frame.get_gemini_client")
    def test_uses_reference_video_and_analysis(
        self,
        mock_get_gemini: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        product_dir: Path,
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
        script_text: str,
    ) -> None:
        ctx, script_id, product_ids = _setup(
            content_store, run_store, product_dir, script_text, default_params, code_version
        )

        analysis_ref = content_store.register_bytes(
            b"# Reference\nSlow push-in, key light camera right.",
            original_name="analysis.md",
            kind="text",
        )
        video_ref = content_store.register_bytes(
            b"\x00\x00\x00\x18ftypmp42fakebytes",
            original_name="ref.mp4",
            kind="video",
            mime="video/mp4",
        )

        mock_gemini = MagicMock()
        mock_get_gemini.return_value = mock_gemini
        active_file = SimpleNamespace(name="files/abc", state="ACTIVE")
        mock_gemini.files.upload.return_value = active_file
        mock_gemini.models.generate_content.side_effect = [
            MagicMock(text="Tight close-up of the right earlobe, hair tucked behind."),
            *_caption_responses(),
            MagicMock(text="Place the product on the right earlobe."),
            _gemini_critique_response(True, [], ""),
            MagicMock(text="Outpaint to a 3:4 portrait."),
        ]

        fake_bytes = _fake_image_bytes()
        mock_fal.subscribe.side_effect = [
            {"images": [{"url": "https://fake/closeup.png"}]},
            {"images": [{"url": "https://fake/composite.png"}]},
            {"images": [{"url": "https://fake/full_frame.png"}]},
        ]
        mock_fal.upload.return_value = "https://fake/uploaded.png"
        mock_httpx.get.return_value = MagicMock(content=fake_bytes)

        result = generate_starting_frame(
            script_id,
            product_ids,
            ctx=ctx,
            max_refinements=3,
            reference_video_content_id=video_ref.content_id,
            reference_analysis_content_id=analysis_ref.content_id,
        )

        assert result.content_id is not None
        run = run_store.get(ctx.run_id)
        attrs = run.steps[STEP_NAME].attributes
        assert attrs["used_reference_video"] is True
        assert attrs["used_reference_analysis"] is True

        # Reference video is uploaded once for Stage A and once for Stage C.
        assert mock_gemini.files.upload.call_count == 2
        assert mock_gemini.files.delete.call_count == 2
        for call in mock_gemini.files.delete.call_args_list:
            assert call.kwargs == {"name": "files/abc"}

        scene_call = mock_gemini.models.generate_content.call_args_list[0]
        scene_contents = scene_call.kwargs["contents"]
        assert scene_contents[0] is active_file
        assert "Slow push-in" in scene_contents[1]

        # The expand-prompt call is the last Gemini call. It receives the
        # close-up composite (Photo 1 caption + bytes), the reference video,
        # and the user prompt — in that order.
        expand_call = mock_gemini.models.generate_content.call_args_list[-1]
        expand_contents = expand_call.kwargs["contents"]
        assert "CLOSE-UP COMPOSITE" in expand_contents[0]
        assert expand_contents[2] is active_file
        assert "Slow push-in" in expand_contents[3]

    @patch("video_generation.steps.generate_starting_frame.httpx")
    @patch("video_generation.steps.generate_starting_frame.fal_client")
    @patch("video_generation.steps.generate_starting_frame.get_gemini_client")
    def test_reuses_cached_captions_from_library(
        self,
        mock_get_gemini: MagicMock,
        mock_fal: MagicMock,
        mock_httpx: MagicMock,
        product_dir: Path,
        content_store: ContentStore,
        run_store: RunStore,
        default_params: RunParams,
        code_version: CodeVersion,
        script_text: str,
    ) -> None:
        """Pre-seeded captions in the content store should bypass Gemini entirely."""
        ctx, script_id, product_ids = _setup(
            content_store, run_store, product_dir, script_text, default_params, code_version
        )

        # Pre-seed captions for every product image (simulating a prior
        # run or scripts/precaption_products.py).
        for cid, caption in zip(product_ids, _PRODUCT_CAPTIONS, strict=True):
            content_store.set_caption(cid, caption, model="seeded")

        mock_gemini = MagicMock()
        mock_get_gemini.return_value = mock_gemini
        # Note: NO caption responses needed -- they all come from the cache.
        mock_gemini.models.generate_content.side_effect = [
            MagicMock(text="Tight close-up scene, unadorned earlobe."),
            MagicMock(text="Place the product on the ear."),
            _gemini_critique_response(True, [], ""),
            MagicMock(text="Outpaint to portrait."),
        ]

        fake_bytes = _fake_image_bytes()
        mock_fal.subscribe.side_effect = [
            {"images": [{"url": "https://fake/closeup.png"}]},
            {"images": [{"url": "https://fake/composite.png"}]},
            {"images": [{"url": "https://fake/full_frame.png"}]},
        ]
        mock_fal.upload.return_value = "https://fake/uploaded.png"
        mock_httpx.get.return_value = MagicMock(content=fake_bytes)

        generate_starting_frame(script_id, product_ids, ctx=ctx, max_refinements=3)

        run = run_store.get(ctx.run_id)
        attrs = run.steps[STEP_NAME].attributes
        assert attrs["captions_generated"] == 0
        assert attrs["captions_reused"] == 3

        # Only 1 closeup-scene + 1 composite + 1 critique + 1 expand = 4 Gemini
        # calls; zero captioning.
        assert mock_gemini.models.generate_content.call_count == 4

        # The cached caption text is still what landed in the recorded snapshot.
        outputs = run.steps[STEP_NAME].outputs
        captions_payload = json.loads(
            content_store.get_bytes(outputs["product_captions"]).decode("utf-8")
        )
        assert {entry["caption"] for entry in captions_payload} == set(_PRODUCT_CAPTIONS)

        # The seeded record's "model" field is preserved (not rewritten).
        for cid in product_ids:
            record = content_store.get_caption_record(cid)
            assert record is not None
            assert record["model"] == "seeded"
