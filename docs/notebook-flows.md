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
│   Analysis + product images → Claude       │
│   → script.md                              │
└──────────┬─────────────────────────────────┘
           │                             │
           ▼                             ▼
┌────────────────────────────────────────────┐
│ Step 3: Generate Starting Frame            │
│   3a. Claude writes scene prompt           │
│   3b. Seedream v4 → base scene (no product)│
│   3c. Gemini writes composite prompt       │
│   3d. Seedream v4.5 Edit → composite       │
│   3e. Gemini critique loop (≤3x)           │
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
| **Services** | Claude (Anthropic) |
| **Input** | Analysis `.md` + product images directory |
| **Output** | `script.md` in output dir |
| **Prompts** | `system/script_writer.txt`, `examples/emulate_reference_script.txt` |

**Processing:**
1. Read the analysis markdown from Step 1's output path
2. `load_product_images()` — find all `.webp/.jpg/.jpeg/.png` in the product directory
3. Build a product description string listing each image filename
4. `load_image_as_base64()` — resize to max 800px, JPEG-encode at quality 90, base64
5. Template-substitute the analysis + product description into the user prompt
6. Build a multimodal Claude message: product images + templated text
7. Claude writes a single continuous shot script that emulates the reference style while showcasing the product
8. Save to `{output_dir}/script.md`

**Result dataclass:** `ScriptResult(script_text, script_path)`

---

### Step 3: Generate Starting Frame

**Module:** `video_generation.steps.generate_starting_frame`

This is the most complex step — three stages with an iterative refinement loop. It uses both Claude (for scene prompt writing) and Gemini (for composite prompt + critique).

| | |
|---|---|
| **Services** | Claude Sonnet (scene prompt), Gemini 3.1 Pro (composite prompt + critique), Seedream v4 (fal), Seedream v4.5 Edit (fal) |
| **Input** | Script text, product images directory |
| **Output** | `starting_frame*.png` + `composite_prompt.txt` + `critique_v*.json` in `output_dir/images/` |
| **Prompts** | `examples/scene_without_product.txt`, `examples/write_composite_prompt.txt`, `examples/critique_composite.txt` |

**Stage 1 — Generate base scene (no product):**
1. Template-substitute the script into `scene_without_product.txt`
2. Claude Sonnet writes an image generation prompt for a scene *without any jewelry*
3. Call **Seedream v4** text-to-image (portrait 4:3) via fal → base scene PNG
4. Download the result image

**Stage 2 — Initial composite (Gemini):**
1. Load all product images, upload them + the scene to fal
2. Send scene + all product photos to **Gemini 3.1 Pro** with `write_composite_prompt.txt` (images passed via `Part.from_bytes()`)
3. Gemini writes an edit prompt describing how to composite the product onto the scene
4. Call **Seedream v4.5 Edit** with the composite prompt + scene URL + product URLs → composited image
5. Save as `starting_frame_v0.png`

**Stage 3 — Iterative refinement (Gemini critique loop, up to 3 rounds):**
1. Send the current composite + all product reference photos to **Gemini 3.1 Pro** with `critique_composite.txt`
2. Gemini returns a structured `CompositesCritique` via JSON schema enforcement: `{acceptable, issues[], correction_prompt}`
3. If `acceptable == true` → stop, use current composite
4. If a `correction_prompt` is provided → call Seedream v4.5 Edit again with the correction
5. Save each iteration as `starting_frame_v{i}.png` and critique as `critique_v{i}.json`
6. Repeat until accepted or max iterations reached
7. Save the final version as `starting_frame_<timestamp>.png`

**Result dataclass:** `StartingFrameResult(frame_path)`

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

All artifacts land in `--output-dir` (default `data/outputs/`):

```
data/outputs/
├── {video_stem}-analysis.md      ← Step 1
├── script.md                     ← Step 2
├── images/
│   ├── starting_frame_v0.png     ← Step 3 (initial composite)
│   ├── starting_frame_v1.png     ← Step 3 (refinement 1)
│   ├── starting_frame_v2.png     ← Step 3 (refinement 2)
│   ├── starting_frame_*.png      ← Step 3 (final)
│   ├── composite_prompt.txt      ← Step 3 (composite edit prompt)
│   ├── critique_v1.json          ← Step 3 (critique round 1)
│   ├── critique_v2.json          ← Step 3 (critique round 2)
│   └── critique_v3.json          ← Step 3 (critique round 3)
└── videos/
    └── video_*.mp4               ← Step 4 (final output)
```

---

## Prompt Template Map

All prompts live in `data/prompts/` and are loaded by `video_generation.prompts.loader`.

### System prompts (`data/prompts/system/`)

| File | Used by |
|---|---|
| `video_analyst.txt` | Step 1 — analyze reference (Gemini) |
| `script_writer.txt` | Step 2 — write script (Claude) |
| `assistant.txt` | General-purpose (not used in pipeline) |

### Example prompts (`data/prompts/examples/`)

| File | Used by |
|---|---|
| `shot_breakdown_request.txt` | Step 1 — user message for video analysis (Gemini) |
| `emulate_reference_script.txt` | Step 2 — templated script request (vars: `$reference_analysis`, `$product_description`) |
| `scene_without_product.txt` | Step 3a — scene prompt request (var: `$script`) (Claude) |
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
| Google Gemini 3.1 Pro | `gemini-3.1-pro-preview` | 1 (video analysis), 3b (composite prompt), 3c (critique) |
| Anthropic Claude Sonnet | `claude-sonnet-4-20250514` | 2 (script), 3a (scene prompt), 4 (motion prompt) |
| fal — Seedream v4 | `fal-ai/bytedance/seedream/v4/text-to-image` | 3a (base scene) |
| fal — Seedream v4.5 Edit | `fal-ai/bytedance/seedream/v4.5/edit` | 3b (composite), 3c (refinement) |
| fal — Kling v2.6 Pro | `fal-ai/kling-video/v2.6/pro/image-to-video` | 4 (default video model) |
| fal — Seedance v1.5 Pro | `fal-ai/bytedance/seedance/v1.5/pro/image-to-video` | 4 (alternative video model) |

---

## API Keys Required

| Service | Environment Variable | Used For |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | Claude (script, scene prompt, motion prompt) |
| Google Gemini | `GEMINI_API_KEY` | Gemini (video analysis, composite prompt, critique) |
| Fal.ai | `FAL_KEY` | Seedream (images), Kling/Seedance (video) |

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

```bash
# Step 1 only
uv run python -m video_generation --step analyze --reference-video ... --product-dir ...

# Step 2 only (requires analysis to exist)
uv run python -m video_generation --step script --reference-analysis ... --product-dir ...

# Step 3 only (requires script.md in output dir)
uv run python -m video_generation --step frame --product-dir ...

# Step 4 only (requires script.md + starting_frame*.png in output dir)
uv run python -m video_generation --step video --product-dir ...
```
