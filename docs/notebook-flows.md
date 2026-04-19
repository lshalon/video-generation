# Notebook & Pipeline Flows: Frames to Final Video

How the codebase turns reference video frames and product images into a finished video advertisement.

## High-Level Data Flow

```
Reference Video (.mp4)          Product Images (.webp/.jpg/.png)
        │                                │
        ▼                                │
┌───────────────────────┐                │
│ Step 1: Analyze       │                │
│   Gemini native video │                │
│   → analysis.md       │                │
└──────────┬────────────┘                │
           │                             │
           ▼                             ▼
┌────────────────────────────────────────────┐
│ Step 2: Write Script                       │
│   Analysis + product images → Gemini 3.1 Pro│
│   → script.md                              │
└──────────┬─────────────────────────────────┘
           │                             │
           ▼                             ▼
┌────────────────────────────────────────────┐
│ Step 3: Generate Starting Frame            │
│   Stage A: Gemini → close-up scene prompt  │
│       (sees reference video if provided)   │
│   Stage A: Nano Banana 2 → 1:1 close-up    │
│       (empty earlobe, no jewellery)        │
│   Stage 1.5: Gemini captions each product  │
│       photo (cached as content sidecar)    │
│   Stage B: Gemini → composite prompt       │
│   Stage B: Nano Banana 2 Edit → composite  │
│       (1:1, on close-up canvas)            │
│   Stage B: Gemini critique loop (≤Nx)      │
│   Stage C: Gemini → outpaint prompt        │
│       (sees reference video if provided)   │
│   Stage C: Nano Banana 2 Edit → 3:4        │
│       (single deterministic outpaint)      │
│   → starting_frame.png                     │
└──────────┬─────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────┐
│ Step 4: Generate Video                     │
│   Claude writes motion prompt              │
│   Kling / Seedance (fal) animates frame    │
│   → video.mp4                              │
└────────────────────────────────────────────┘
```

---

## Notebooks

The repo has three notebooks under `notebooks/`. They are **supporting tools**, not a required sequential chain — the authoritative end-to-end flow is the CLI (`python -m video_generation`). No notebook passes artifacts to another notebook.

| Notebook | Purpose | Pipeline step covered |
|---|---|---|
| `video_analysis.ipynb` | Interactive reference analysis | Step 1 only |
| `scratch.ipynb` | Ad-hoc experiments and prompt exploration | None (exploratory) |
| `run_tests.ipynb` | Runs pytest, ruff, mypy from notebook | None (QA) |

### `video_analysis.ipynb` — Reference Analysis (Step 1)

This is the only notebook that implements a pipeline step. It reproduces Step 1 interactively using Claude (the notebook predates the Gemini migration — the CLI now uses Gemini native video input instead).

**Inputs:**
- Hard-coded path: `../data/inputs/jewelry/reference/yurman-full-shot.mp4`
- Config: `MODEL = "claude-sonnet-4-20250514"`, `NUM_FRAMES = 20`

**Flow:**
1. Import `cv2`, `numpy`, `base64` and helpers from `video_generation` (`load_prompt`, `load_system_prompt`, `get_anthropic_client`, `load_env`)
2. Define `extract_frames(video_path, num_frames)` — uses OpenCV to extract evenly-spaced frames
3. Define `frame_to_base64(frame, max_size=1024)` — resize + JPEG-encode + base64
4. Load prompts: system = `video_analyst.txt`, user = `shot_breakdown_request.txt`
5. Build multimodal message: all frame images + text prompt
6. Call `client.messages.create(...)` with Claude
7. Display the analysis markdown inline
8. Save result to `../data/inputs/jewelry/reference/yurman-full-shot-analysis.md`

**Output:** `{video_stem}-analysis.md` — a cinematography breakdown (shot types, camera movement, lighting, pacing).

> **Note:** The CLI pipeline now uses Gemini with native video upload (no frame extraction needed). This notebook still uses the legacy Claude-based approach with extracted frames.

### `scratch.ipynb` — Exploration

A sandbox for experimenting with the `video_generation` package. Demonstrates:
- `list_prompts()`, `load_prompt()`, `save_prompt()`
- `display_image()`, `display_video()`, `save_image()`, `save_video()`
- `load_env()`, `get_api_key()`, `check_api_keys()`

No pipeline artifacts are produced.

### `run_tests.ipynb` — QA

Runs shell commands via `subprocess` with `cwd=".."` (repo root):
- `pytest` (all tests or a specific `TEST_FILE`)
- `pytest --cov`
- `ruff check`
- `mypy`

---

## Full Pipeline Steps (CLI)

All steps live under `src/video_generation/steps/` and are orchestrated by `pipeline.py`.

### Step 1: Analyze Reference

**Module:** `video_generation.steps.analyze_reference`

| | |
|---|---|
| **Services** | Gemini (Google, via `google-genai` SDK) |
| **Input** | Reference video `.mp4` |
| **Output** | `{video_stem}-analysis.md` in output dir |
| **Prompts** | `system/video_analyst.txt`, `examples/shot_breakdown_request.txt` |

**Processing:**
1. Upload the video to Gemini File API (`client.files.upload()`)
2. Poll until file state is `ACTIVE` (Gemini processes video at 1 FPS + audio)
3. Send the video file + system prompt + shot breakdown request to Gemini
4. Gemini returns a markdown analysis of cinematography, lighting, camera movement, pacing — with access to the full video stream including motion and audio, not just static frame samples
5. Clean up the uploaded file
6. Save to `{output_dir}/{video_stem}-analysis.md`

**Result dataclass:** `AnalysisResult(analysis_text, output_path)`

---

### Step 2: Write Script

**Module:** `video_generation.steps.write_script`

| | |
|---|---|
| **Services** | Gemini 3.1 Pro (Google) |
| **Input** | Reference analysis content id + product image content ids |
| **Output** | `script.md` (recorded as content + convenience copy under `runs/<run_id>/steps/script/`) |
| **Prompts** | `system/script_writer.txt`, `examples/emulate_reference_script.txt` |

**Processing:**
1. Read the analysis markdown via `ctx.content_store.get_bytes(...)`.
2. For each product image content id, fetch the bytes, downscale to ≤800px,
   JPEG-encode (quality 90), and wrap as a `types.Part.from_bytes(...)`.
3. Template-substitute the analysis + product description into
   `examples/emulate_reference_script.txt`.
4. Send `[image_part, image_part, ..., text_prompt]` to Gemini 3.1 Pro with
   `thinking_level=LOW`. Output length is constrained by the prompt itself
   (`emulate_reference_script.txt` asks for ~250–500 words under fixed
   headings); we don't set `max_output_tokens`.
5. Record the script as `script` (kind=text, mime=text/markdown).

**Result dataclass:** `ScriptResult(script_text, script_path, content_id)`

---

### Step 3: Generate Starting Frame

**Module:** `video_generation.steps.generate_starting_frame`

The most complex step — a *close-up then outpaint* flow with three internal
stages plus a caption pre-step. Gemini 3.1 Pro drives every text-generation
call (close-up scene prompt, captions, composite prompt, critique, expand
prompt); Nano Banana 2 handles every image generation/edit.

> **Why the close-up?** A ~4 mm earring on a 3:4 portrait is only a few
> dozen pixels tall, even at NB2 2K. The editor would not converge on the
> right *size* relative to the lobe — and neither would the critique loop,
> however aggressive its prompt. By compositing onto a 1:1 close-up first,
> the earring naturally fills 5–10% of the canvas, the editor's bias works
> in our favour, and the critique loop has room to manoeuvre. The final
> 3:4 portrait is then produced in a single deterministic outpaint that is
> instructed to PRESERVE the close-up region exactly.

| | |
|---|---|
| **Services** | Gemini 3.1 Pro (close-up scene prompt, captions, composite prompt, critique, expand prompt), Nano Banana 2 + Nano Banana 2 Edit (fal.ai) |
| **Input** | Script content id, product image content ids, optional reference video + analysis content ids |
| **Output** | `closeup_scene_prompt.txt`, `closeup_scene.png`, `product_captions.json`, `composite_prompt.txt`, `closeup_composite_v0..N.png`, `critique_v{i}.json`, `closeup_composite_final.png`, `expand_prompt.txt`, `starting_frame.png` (final 3:4 outpainted) |
| **Prompts** | `examples/closeup_scene_without_product.txt`, `examples/caption_product_image.txt`, `examples/write_composite_prompt.txt`, `examples/critique_composite.txt`, `examples/expand_to_full_frame.txt` |

**Stage A — Generate close-up scene (no product):**
1. Template-substitute the script + reference analysis into
   `closeup_scene_without_product.txt`.
2. If a `reference_video_content_id` is provided, materialize the bytes into
   a temp dir, upload to the Gemini File API, and poll until `state=ACTIVE`.
3. Send `[video_file, closeup_scene_request_text]` (or just text if no
   video) to Gemini 3.1 Pro with `SCENE_PROMPT_SYSTEM` and
   `thinking_level=LOW`. The prompt asks for one dense paragraph describing
   an extreme close-up of the ear; no `max_output_tokens` cap. Gemini
   matches the reference's lighting, skin tone, and styling.
4. Delete the uploaded Gemini file.
5. Call **Nano Banana 2** (`fal-ai/nano-banana-2`, constant
   `BASE_SCENE_TEXT_TO_IMAGE`) text-to-image with `aspect_ratio="1:1"`,
   `resolution=NANO_BANANA_RESOLUTION` (default `"2K"`),
   `output_format="png"`.
6. Download the resulting PNG. Recorded as `closeup_scene` in the manifest.

**Stage 1.5 — Caption every product image (content-tied, cached forever):**
1. For each unique product `content_id`, check the content store for an
   existing `library/<aa>/<sha>.caption.json` sidecar.
2. If present, reuse it (no LLM call).
3. Otherwise, send the image bytes + `caption_product_image.txt` to Gemini
   3.1 Pro at `thinking_level=LOW` and persist the result back to the
   content store via `ContentStore.set_caption(...)`.
4. Record the resolved `{content_id: caption}` mapping as
   `product_captions.json` in the run manifest. Per-run attributes
   `captions_generated` and `captions_reused` summarise cache hit rate.

**Stage B — Composite earring onto close-up + critique loop:**
1. Build a Gemini multimodal `contents` list with explicit per-image
   captions: `["Photo 1 (BASE SCENE — close-up):", closeup_part, "Photo 2
   (<gemini caption>):", product_part, ..., write_composite_prompt_text]`.
2. Send to Gemini 3.1 Pro with `thinking_level=HIGH`. The composite prompt
   template is aware that the canvas is a 1:1 close-up, so it asks for the
   correct *proportion to the lobe* (matching the on-model reference photo)
   rather than a small visual size in the frame.
3. Upload the close-up + each product image to fal.
4. Call **Nano Banana 2 Edit** (`fal-ai/nano-banana-2/edit`) with
   `prompt=composite_prompt`, `image_urls=[closeup_url, *product_urls]`,
   `aspect_ratio="1:1"`, `resolution=NANO_BANANA_RESOLUTION`. Recorded as
   `closeup_composite_v0`.
5. **Critique loop** (off when `--max-refinements 0`):
   - Send the current composite + all captioned product references to
     Gemini 3.1 Pro with `critique_composite.txt`, structured-output JSON
     (`response_schema=CompositesCritique`), `thinking_level=LOW`.
   - Gemini returns `{acceptable, issues[], correction_prompt}`. The
     critique prompt tells Gemini that its `correction_prompt` will be sent
     to the editor with the same set of reference images in the same order,
     so corrections refer to images by position ("the on-model reference
     photo", "the front view") rather than describing size in the abstract.
   - If `acceptable == true`, exit the loop.
   - Otherwise, run NB2 Edit again at 1:1 with
     `image_urls=[current_composite, *product_urls]` so the editor still has
     the size + design ground truths in-context. Record
     `closeup_composite_v{i}` and `critique_v{i}`.
   - Loop up to `--max-refinements` times.
6. The accepted (or last-seen) close-up composite is recorded as
   `closeup_composite_final`.

**Stage C — Outpaint close-up to 3:4 portrait (single shot, no critique):**
1. Build a Gemini multimodal `contents` list:
   `["Photo 1 (CLOSE-UP COMPOSITE — must be preserved exactly):",
   closeup_composite_part, <reference_video?>, expand_request_text]`.
2. Send to Gemini 3.1 Pro with `EXPAND_PROMPT_SYSTEM` and
   `thinking_level=HIGH`. Gemini writes one outpaint instruction grounded in
   `expand_to_full_frame.txt` — telling the editor to PRESERVE the
   close-up region pixel-for-pixel and INVENT only the rest of the head,
   hair, shoulders, and background to match the reference framing. Recorded
   as `expand_prompt`.
3. Upload the close-up composite to fal as the only input image.
4. Call NB2 Edit with `prompt=expand_prompt`, `image_urls=[closeup_url]`,
   `aspect_ratio="3:4"`, `resolution=NANO_BANANA_RESOLUTION`.
5. The result is recorded as `starting_frame` — the same name the video
   step (Step 4) consumes.

**Result dataclass:** `StartingFrameResult(frame_path, content_id, intermediate_content_ids)`

---

### Step 4: Generate Video

**Module:** `video_generation.steps.generate_video`

| | |
|---|---|
| **Services** | Claude (Anthropic), Kling v2.6 Pro or Seedance v1.5 Pro (fal) |
| **Input** | Starting frame `.png`, script text |
| **Output** | `video_*.mp4` in `output_dir/videos/` |
| **Prompts** | `examples/video_motion_prompt.txt` |

**Processing:**
1. Read the starting frame from Step 3's output path
2. Template-substitute the script + a frame description into `video_motion_prompt.txt`
3. Compress the frame for Claude (max 1024px, JPEG quality 85)
4. Send frame image + motion request to Claude → short 1-2 sentence motion prompt
5. Compress the frame for upload (max 1920px, JPEG quality 95)
6. Upload to fal via `fal_client.upload()`
7. Build model-specific arguments:
   - **Kling**: uses `start_image_url`, adds a `negative_prompt`
   - **Seedance**: uses `image_url`, no negative prompt
8. Call the video model via `fal_client.subscribe()` → video URL
9. Download the MP4
10. Save to `{output_dir}/videos/video_<timestamp>.mp4`

**Result dataclass:** `VideoResult(video_path)`

---

## Artifact Map

Artifacts are stored content-addressed under `<storage_root>/library/` and
mirrored into the run's convenience-copy tree at
`<storage_root>/runs/<run_id>/steps/`. For `--storage local:./data` (default):

```
data/
├── library/<aa>/<sha256>{.ext,.meta.json,.caption.json}   # canonical blobs + caption sidecars
└── runs/<run_id>/
    ├── manifest.json                           # RunRecord
    └── steps/
        ├── analyze/analysis.md                 ← Step 1
        ├── script/script.md                    ← Step 2
        ├── frame/                              ← Step 3 (close-up then outpaint)
        │   ├── closeup_scene_prompt.txt        # Stage A prompt
        │   ├── closeup_scene.png               # Stage A image (1:1, no jewellery)
        │   ├── product_captions.json           # Stage 1.5 (resolved per-image captions)
        │   ├── composite_prompt.txt            # Stage B prompt
        │   ├── closeup_composite_v0.png        # Stage B initial composite (1:1)
        │   ├── critique_v1.json                # Stage B critique (if --max-refinements > 0)
        │   ├── closeup_composite_v1.png        # Stage B refined composite (1:1)
        │   ├── ...
        │   ├── closeup_composite_final.png     # Stage B accepted close-up
        │   ├── expand_prompt.txt               # Stage C outpaint prompt
        │   └── starting_frame.png              # Stage C outpainted final 3:4 portrait
        └── video/                              ← Step 4
            ├── motion_prompt.txt
            └── video.mp4
```

The legacy `--output-dir` (default `data/outputs/`) is kept for backward
compatibility but no longer written to by the pipeline. See
[storage-and-runs.md](storage-and-runs.md) for the manifest schema and
content-store details.

---

## Prompt Template Map

All prompts live in `data/prompts/` and are loaded by `video_generation.prompts.loader`.

### System prompts (`data/prompts/system/`)

| File | Used by |
|---|---|
| `video_analyst.txt` | Step 1 — analyze reference (Gemini) |
| `script_writer.txt` | Step 2 — write script (Gemini) |
| `assistant.txt` | General-purpose (not used in pipeline) |

### Example prompts (`data/prompts/examples/`)

| File | Used by |
|---|---|
| `shot_breakdown_request.txt` | Step 1 — user message for video analysis (Gemini) |
| `emulate_reference_script.txt` | Step 2 — templated script request (vars: `$reference_analysis`, `$product_description`) (Gemini) |
| `closeup_scene_without_product.txt` | Step 3 / Stage A — close-up scene prompt request (vars: `$script`, `$reference_analysis`) (Gemini, also receives the reference video binary when available) |
| `caption_product_image.txt` | Step 3 / Stage 1.5 — one-sentence content-tied caption per product image, results cached as `library/<aa>/<sha>.caption.json` (Gemini) |
| `write_composite_prompt.txt` | Step 3 / Stage B — composite prompt from close-up + captioned product images (Gemini) |
| `critique_composite.txt` | Step 3 / Stage B critique — structured critique of close-up composite (Gemini) |
| `expand_to_full_frame.txt` | Step 3 / Stage C — outpaint instruction expanding close-up to 3:4 portrait (vars: `$reference_analysis`) (Gemini, also receives the reference video binary when available) |
| `video_motion_prompt.txt` | Step 4 — motion prompt request (vars: `$script`, `$starting_frame_description`) (Claude) |
| `starting_frame_prompt.txt` | Legacy (not used in pipeline) |
| `locate_ear_region.txt` | Legacy (not used in pipeline) |
| `analyze_product.txt` | Not used in pipeline |
| `image_generation.txt` | Not used in pipeline (example) |
| `video_generation.txt` | Not used in pipeline (example) |

---

## External Services

| Service | Endpoint / Model | Steps |
|---|---|---|
| Google Gemini 3.1 Pro | `gemini-3.1-pro-preview` | 1 (video analysis), 2 (script), 3 / Stage A (close-up scene prompt, w/ reference video), 3 / Stage 1.5 (captions), 3 / Stage B (composite prompt + critique), 3 / Stage C (outpaint prompt, w/ reference video) |
| Anthropic Claude Sonnet | `claude-sonnet-4-20250514` | 4 (motion prompt only) |
| fal — Nano Banana 2 | `fal-ai/nano-banana-2` | 3 / Stage A (1:1 close-up base scene) |
| fal — Nano Banana 2 Edit | `fal-ai/nano-banana-2/edit` | 3 / Stage B (1:1 composite + refinement), 3 / Stage C (3:4 outpaint) |
| fal — Kling v2.6 Pro | `fal-ai/kling-video/v2.6/pro/image-to-video` | 4 (default video model) |
| fal — Seedance v1.5 Pro | `fal-ai/bytedance/seedance/v1.5/pro/image-to-video` | 4 (alternative video model) |

> Older `fal-ai/bytedance/seedream/v4{,.5}/...` endpoints are still supported
> via `--edit-model` (the `_run_edit` helper branches on the model id and
> swaps `aspect_ratio` for Seedream's `image_size` parameter).

---

## API Keys Required

| Service | Environment Variable | Used For |
|---|---|---|
| Google Gemini | `GEMINI_API_KEY` | Gemini (analysis, script, scene prompt, composite prompt, critique) |
| Anthropic | `ANTHROPIC_API_KEY` | Claude (video motion prompt only) |
| Fal.ai | `FAL_KEY` | Nano Banana 2 (images), Kling/Seedance (video) |

---

## Running the Pipeline

### Full pipeline (with reference video)

```bash
uv run python -m video_generation \
  --product-dir data/inputs/jewelry/product \
  --reference-video data/inputs/jewelry/reference/yurman-full-shot.mp4
```

### Full pipeline (skip analysis, reuse existing)

```bash
uv run python -m video_generation \
  --product-dir data/inputs/jewelry/product \
  --reference-analysis data/inputs/jewelry/reference/yurman-full-shot-analysis.md
```

### Override models

```bash
uv run python -m video_generation \
  --product-dir data/inputs/jewelry/product \
  --reference-video data/inputs/jewelry/reference/yurman-full-shot.mp4 \
  --gemini-model gemini-3-flash-preview \
  --claude-model claude-sonnet-4-20250514
```

### Single steps

`--step` runs one named step. Any missing upstream steps for the same
`(inputs, params, code_version)` triple are run automatically; previously
completed steps are short-circuited from the manifest, so re-invoking a
single step on a complete run does **not** make any external API calls.

```bash
# Step 1 only
uv run python -m video_generation --step analyze \
  --product-dir data/inputs/jewelry/product \
  --reference-video data/inputs/jewelry/reference/yurman-full-shot.mp4

# Step 2 only — auto-runs Step 1 if needed for this run
uv run python -m video_generation --step script \
  --product-dir data/inputs/jewelry/product \
  --reference-video data/inputs/jewelry/reference/yurman-full-shot.mp4

# Step 3 only — auto-runs Steps 1+2 if needed
uv run python -m video_generation --step frame \
  --product-dir data/inputs/jewelry/product \
  --reference-video data/inputs/jewelry/reference/yurman-full-shot.mp4

# Step 4 only — auto-runs everything upstream that's missing
uv run python -m video_generation --step video \
  --product-dir data/inputs/jewelry/product \
  --reference-video data/inputs/jewelry/reference/yurman-full-shot.mp4
```

To force a step to re-execute, change something that participates in
`run_id` (e.g. pass `--variant rerun-1`) or pass `--max-refinements 0` to
skip the critique loop.
