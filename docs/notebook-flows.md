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
│   3a. Gemini writes scene prompt           │
│       (sees reference video if provided)   │
│   3b. Nano Banana 2 → base scene (no product)│
│   3c. Gemini writes composite prompt       │
│   3d. Nano Banana 2 Edit → composite       │
│   3e. Gemini critique loop (≤3x, optional) │
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
   `thinking_level=LOW` and `max_output_tokens=8192`.
5. Record the script as `script` (kind=text, mime=text/markdown).

**Result dataclass:** `ScriptResult(script_text, script_path, content_id)`

---

### Step 3: Generate Starting Frame

**Module:** `video_generation.steps.generate_starting_frame`

The most complex step — three stages with an optional iterative refinement
loop. Gemini 3.1 Pro drives all three text-generation calls (scene prompt,
composite prompt, critique); Nano Banana 2 handles both image generations.

| | |
|---|---|
| **Services** | Gemini 3.1 Pro (scene prompt, composite prompt, critique), Nano Banana 2 + Nano Banana 2 Edit (fal.ai) |
| **Input** | Script content id, product image content ids, optional reference video + analysis content ids |
| **Output** | `scene_prompt.txt`, `base_scene.png`, `composite_prompt.txt`, `starting_frame_v0.png`, `critique_v{i}.json`, `starting_frame_v{i}.png`, `starting_frame.png` (alias of the final version) |
| **Prompts** | `examples/scene_without_product.txt`, `examples/write_composite_prompt.txt`, `examples/critique_composite.txt` |

**Stage 1 — Generate base scene (no product):**
1. Template-substitute the script + reference analysis into
   `scene_without_product.txt`.
2. If a `reference_video_content_id` is provided, materialize the bytes into
   a temp dir, upload to the Gemini File API, and poll until `state=ACTIVE`.
3. Send `[video_file, scene_request_text]` (or just text if no video) to
   Gemini 3.1 Pro with `SCENE_PROMPT_SYSTEM`,
   `thinking_level=LOW`, and `max_output_tokens=4096`. Gemini matches the
   reference's lens, lighting, framing, and pose.
4. Delete the uploaded Gemini file.
5. Call **Nano Banana 2** (`fal-ai/nano-banana-2`) text-to-image with
   `aspect_ratio="3:4"`, `resolution="1K"`, `output_format="png"`.
6. Download the resulting PNG.

**Stage 2 — Initial composite (Gemini + Nano Banana 2 Edit):**
1. Build a Gemini multimodal `contents` list: `[base_scene_png_part,
   product_png_part, ..., write_composite_prompt_text]`.
2. Send to Gemini 3.1 Pro with `thinking_level=LOW` and
   `max_output_tokens=4096`. Gemini writes a precise, scale-aware edit
   prompt with explicit DO / DO NOT lists.
3. Upload the base scene + each product image to fal.
4. Call **Nano Banana 2 Edit** (`fal-ai/nano-banana-2/edit`) with
   `prompt=composite_prompt`, `image_urls=[scene_url, *product_urls]`,
   `aspect_ratio="3:4"`. (The `_run_edit()` helper branches on model id so
   passing a Seedream `--edit-model` still works with `image_size`.)
5. Record as `starting_frame_v0` (and as `starting_frame` if
   `--max-refinements 0`).

**Stage 3 — Iterative refinement (optional, off when `--max-refinements 0`):**
1. Send the current composite + all product reference photos to Gemini 3.1
   Pro with `critique_composite.txt`, structured-output JSON
   (`response_schema=CompositesCritique`), and `thinking_level=LOW`.
2. Gemini returns `{acceptable, issues[], correction_prompt}`.
3. If `acceptable == true`, stop.
4. Otherwise, call Nano Banana 2 Edit again with the correction prompt
   applied to the *current* composite.
5. Record `critique_v{i}.json` and `starting_frame_v{i}.png` for each
   iteration. The final version is also aliased as `starting_frame`.

> **Known caveat:** with the existing critic prompt (`critique_composite.txt`,
> "BE HARSH"), the loop tends to demand cumulative shrinkage and degrades
> Nano Banana 2 Edit's already-good first composite. In practice
> `--max-refinements 0` produces better starting frames; the critic infra is
> kept in place for future scoring/observability use.

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
├── library/<aa>/<sha256>{.ext,.meta.json}     # canonical content blobs
└── runs/<run_id>/
    ├── manifest.json                           # RunRecord
    └── steps/
        ├── analyze/analysis.md                 ← Step 1
        ├── script/script.md                    ← Step 2
        ├── frame/                              ← Step 3
        │   ├── scene_prompt.txt
        │   ├── base_scene.png
        │   ├── composite_prompt.txt
        │   ├── starting_frame_v0.png           # initial composite
        │   ├── critique_v1.json                # only if --max-refinements > 0
        │   ├── starting_frame_v1.png
        │   ├── ...
        │   └── starting_frame.png              # alias of the final version
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
| `scene_without_product.txt` | Step 3a — scene prompt request (vars: `$script`, `$reference_analysis`) (Gemini, also receives the reference video binary when available) |
| `write_composite_prompt.txt` | Step 3b — composite prompt from scene + products (Gemini) |
| `critique_composite.txt` | Step 3c — structured critique of composite (Gemini) |
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
| Google Gemini 3.1 Pro | `gemini-3.1-pro-preview` | 1 (video analysis), 2 (script), 3a (scene prompt, w/ reference video), 3b (composite prompt), 3c (critique) |
| Anthropic Claude Sonnet | `claude-sonnet-4-20250514` | 4 (motion prompt only) |
| fal — Nano Banana 2 | `fal-ai/nano-banana-2` | 3a (base scene) |
| fal — Nano Banana 2 Edit | `fal-ai/nano-banana-2/edit` | 3b (composite), 3c (refinement) |
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
