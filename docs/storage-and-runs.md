# Storage, Content IDs, and Runs

This document covers how the pipeline organizes its inputs and outputs:
content-addressed storage, the `RunRecord` manifest, and how to swap the
local filesystem for Google Drive without changing pipeline code.

The companion module is `video_generation.store` (see
[`src/video_generation/store/`](../src/video_generation/store)).

---

## Mental model

Three orthogonal concepts:

1. **Content** — an immutable byte blob identified by `sha256(bytes)`.
   Has no semantic meaning on its own; it is just bytes plus a small metadata
   sidecar (`mime`, `kind`, `original_name`, `produced_by`).
2. **Role** — the *semantic* slot a piece of content fills in a particular
   step or run. Examples: `reference_video`, `product_images[0]`,
   `script`, `starting_frame`, `composite_prompt`. The same content can play
   different roles in different runs.
3. **Run** — a deterministic execution record:
   `(inputs by role) + (params) + (code version)`. Same triple → same
   `run_id` → resumable, idempotent re-execution.

The storage layer underneath is a pluggable byte-level KV store
(`Storage` Protocol) so the same logic works against the local filesystem
or Google Drive.

---

## On-disk layout

For `--storage local:./data`:

```
data/
├── library/                                 # Content-addressed blobs
│   └── <aa>/                                # First two hex chars of sha256
│       ├── <sha256><ext>                    # The bytes
│       └── <sha256>.meta.json               # ContentMeta sidecar
└── runs/
    └── <run_id>/
        ├── manifest.json                    # RunRecord
        └── steps/                           # Convenience copies (human-browsable)
            ├── analyze/analysis.md
            ├── script/script.md
            ├── frame/{scene_prompt.txt, base_scene.png,
            │          composite_prompt.txt, starting_frame_v0.png,
            │          critique_v1.json, starting_frame_v1.png, …,
            │          starting_frame.png}
            └── video/{motion_prompt.txt, video.mp4}
```

The bytes under `runs/<run_id>/steps/` are *copies* of the canonical content
in `library/`. They exist so a developer can browse a run directory without
resolving content ids; deleting them is safe — the manifest holds the
authoritative content id pointers.

For `--storage gdrive:<folder-id>`, the same key namespace is mirrored as a
nested folder hierarchy under the configured root folder in Drive.

---

## Content ids

A `content_id` is the lowercase hex sha256 of the blob. Identical bytes
register exactly once; re-registering returns the existing
[`ContentRef`](../src/video_generation/store/models.py).

`ContentMeta` (`library/<aa>/<sha>.meta.json`):

```json
{
  "sha256": "ab12...",
  "size": 184320,
  "mime": "image/png",
  "kind": "image",
  "original_name": "starting_frame_v2.png",
  "registered_at": "2026-04-18T16:24:17Z",
  "produced_by": {
    "run_id": "9f3e2c4a8d1b7e60",
    "step": "frame",
    "output_name": "starting_frame_v2"
  }
}
```

- `kind` is one of `image`, `video`, `text`, `json`, `blob`. Inferred from
  `mime` and `original_name` if not supplied.
- `produced_by` is `None` for inputs registered from the CLI; for step
  outputs it points back to the run + step that produced the blob.

API (see `ContentStore` in
[`src/video_generation/store/content.py`](../src/video_generation/store/content.py)):

| Method | Purpose |
|---|---|
| `register_bytes(data, kind=, original_name=, mime=, produced_by=)` | Register raw bytes; returns `ContentRef`. Idempotent on sha. |
| `register_path(path, kind=, produced_by=)` | Convenience for files on disk. |
| `get_bytes(content_id)` | Read the blob's bytes. |
| `get_meta(content_id)` | Read just the metadata sidecar. |
| `get_ref(content_id)` | Combined handle. |
| `materialize(content_id, dest)` | Write blob bytes to `dest` (file or directory). Useful when handing bytes to libraries that need a real filesystem path (e.g. `fal_client.upload_file`). |

---

## Run records

A `RunRecord` is the manifest for one execution. It is persisted as
`runs/<run_id>/manifest.json` and updated atomically each time a step
records output.

### `run_id` derivation

```python
payload = {
  "inputs": inputs.to_dict(),
  "params": params.to_dict(),
  "code_version": code_version.to_dict(),
}
run_id = sha256(canonical_json(payload))[:16]
```

Same triple → same `run_id`. To intentionally fork the same triple, set
`params.variant` to a unique tag (`--variant my-experiment`).

### `code_version`

Captured by `current_code_version()`:

```json
{
  "git_sha": "8c1e9f4b…",
  "git_dirty": true
}
```

- `git_sha` is `git rev-parse HEAD`.
- `git_dirty` is `True` whenever `git status --porcelain` is non-empty.

A dirty checkout participates in the `run_id` so two runs against the same
inputs and params from a dirty tree are still considered the same run (they
share `git_sha=… git_dirty=true`). To get a fresh run id, commit, or pass
`--variant`.

### Manifest schema

```json
{
  "run_id": "9f3e2c4a8d1b7e60",
  "status": "completed",
  "started_at": "2026-04-18T16:24:00Z",
  "finished_at": "2026-04-18T16:32:11Z",
  "code_version": { "git_sha": "8c1e9f4…", "git_dirty": false },
  "params": {
    "gemini_model": "gemini-3.1-pro-preview",
    "claude_model": "claude-sonnet-4-20250514",
    "edit_model": "fal-ai/nano-banana-2/edit",
    "video_model": "fal-ai/kling-video/v2.6/pro/image-to-video",
    "video_duration": "5",
    "max_refinements": 3,
    "variant": ""
  },
  "inputs": {
    "product_images": ["ab12…", "cd34…"],
    "reference_video": "ef56…",
    "reference_analysis": null
  },
  "steps": {
    "analyze": {
      "name": "analyze",
      "status": "completed",
      "started_at": "...",
      "finished_at": "...",
      "inputs": { "reference_video": "ef56…" },
      "outputs": { "analysis": "11aa…" },
      "attributes": { "model": "gemini-3.1-pro-preview" }
    },
    "script": { "...": "..." },
    "frame":  {
      "outputs": {
        "scene_prompt": "...",
        "base_scene": "...",
        "composite_prompt": "...",
        "starting_frame_v0": "...",
        "critique_v1": "...",
        "starting_frame_v1": "...",
        "starting_frame": "..."
      },
      "attributes": {
        "accepted": true,
        "refinements_done": 1,
        "final_critique": { "...": "..." }
      }
    },
    "video":  { "...": "..." }
  }
}
```

Step records are independent: an output `outputs[name]` is a `content_id`
pointer; the bytes always live in `library/`.

---

## Idempotency and resuming

`run_pipeline` follows this loop per step:

1. Look up the `StepRecord` in the loaded `RunRecord`.
2. If `status == "completed"` and the expected outputs are all present,
   skip the step entirely (no API calls).
3. Otherwise, build a `StepContext`, call the step function, persist its
   outputs, and mark the step completed.

Practically:

- Killing the process mid-run and re-invoking `run_pipeline` with the same
  flags resumes from the last completed step.
- `--step frame` will transparently re-run any missing upstream steps for
  the same triple, then run `frame` and stop.
- To force a re-run of a single step, delete its entry from the manifest
  (or change a parameter so the run_id changes).

---

## `StepContext`: the per-step facade

Every step receives a [`StepContext`](../src/video_generation/store/context.py)
rather than the raw stores. This keeps step code focused on the actual
transformation and centralizes recording logic.

```python
def write_script(
    reference_analysis_content_id: str,
    product_image_content_ids: list[str],
    ctx: StepContext,
    gemini_model: str = "gemini-3.1-pro-preview",
) -> ScriptResult:
    analysis_md = ctx.content_store.get_bytes(reference_analysis_content_id).decode()
    images = [ctx.content_store.get_bytes(cid) for cid in product_image_content_ids]

    script_md = call_gemini(gemini_model, analysis_md, images)

    content_id = ctx.record(
        name="script",
        data=script_md.encode(),
        original_name="script.md",
        mime="text/markdown",
        kind="text",
    )
    ctx.set_attribute("gemini_model", gemini_model)
    output_path = ctx.run_store.local_step_output_path(ctx.run_id, "script", "script")
    return ScriptResult(script_text=script_md, script_path=output_path or Path(""), content_id=content_id)
```

`StepContext` methods:

| Method | Purpose |
|---|---|
| `record(output_name, data=, path=, ...)` | Register bytes (or a file) as content **and** attach the resulting `content_id` to `step.outputs[output_name]`. Also writes the convenience copy. |
| `set_attribute(name, value)` | JSON-friendly metadata on the step (model name, refinements done, critique payloads, …). |
| `materialize_input(content_id, dest_dir)` | Write a content id's bytes to a temp/working dir for tools that need a path (e.g. `fal_client.upload_file`). |
| `begin()` / `end(status=...)` | Mark the step as `running` / `completed` (or `failed`) in the manifest. Bracketing the step body with these is what makes the resume-on-restart logic work. |

For the local convenience-copy path of a recorded output, ask the `RunStore`
directly (returns `None` for non-local backends):

```python
ctx.run_store.local_step_output_path(ctx.run_id, "script", "script")
```

---

## Storage backends

### `LocalStorage`

Default backend. Stores bytes under a single root directory using atomic
`*.tmp` + `os.replace` writes. `url(key)` returns a `file://` URL.

```bash
uv run python -m video_generation \
  --storage local:./data \
  --product-dir data/inputs/jewelry/product \
  --reference-video data/inputs/jewelry/reference/yurman-full-shot.mp4
```

### `GoogleDriveStorage`

Mirrors the same key namespace as folders/files inside a Drive folder.
Authentication is service-account based.

#### One-time setup

1. In Google Cloud Console, create (or pick) a project and enable the
   **Google Drive API**.
2. Create a **service account**, generate a JSON key, and download it.
3. In Drive, create a folder that will host the library + runs. Share the
   folder with the service account's email (`...@...iam.gserviceaccount.com`)
   as **Editor**. Note the folder id from the URL
   (`https://drive.google.com/drive/folders/<FOLDER_ID>`).
4. Set the credentials env var:

   ```bash
   export GOOGLE_DRIVE_CREDENTIALS=/abs/path/to/service-account.json
   ```

5. Run with the gdrive spec:

   ```bash
   uv run python -m video_generation \
     --storage gdrive:<FOLDER_ID> \
     --product-dir data/inputs/jewelry/product \
     --reference-video data/inputs/jewelry/reference/yurman-full-shot.mp4
   ```

The Drive layout under `<FOLDER_ID>` looks like:

```
<FOLDER_ID>/
├── library/<aa>/<sha256><ext>     (file)
├── library/<aa>/<sha256>.meta.json
└── runs/<run_id>/manifest.json
└── runs/<run_id>/steps/<step>/<output_name><ext>
```

`url(key)` returns `https://drive.google.com/file/d/<file-id>/view`.

#### Caveats

- Each `put` does a Drive folder lookup per path segment (cached
  per-process). For high-throughput workloads, prefer `LocalStorage` and
  sync to Drive out-of-band.
- `materialize(content_id, ...)` works against any backend — bytes are
  fetched from storage and written to the local destination.
- Convenience copies under `runs/<run_id>/steps/...` are uploaded as Drive
  files, so they consume a small amount of extra space relative to the
  canonical content in `library/`.

### Adding a new backend

Implement the `Storage` protocol
(`put`, `get`, `exists`, `list`, `delete`, `url`) and register a scheme in
[`make_storage`](../src/video_generation/store/factory.py). Nothing else
in the pipeline needs to change.

---

## CLI cheat sheet

```bash
uv run python -m video_generation \
  --product-dir data/inputs/jewelry/product \
  --reference-video data/inputs/jewelry/reference/yurman-full-shot.mp4 \
  --storage local:./data \
  --variant ""
```

- `--storage local:<path>` (default `local:./data`) or `--storage gdrive:<folder-id>`
- `--run-id <id>` to inspect or resume a specific run
- `--variant <tag>` to fork a new run from the same `(inputs, params, code)`
- `--step {analyze|script|frame|video}` to run a single step (missing
  upstream steps run automatically; cached steps are skipped)

---

## Migration

To move existing files from `data/inputs/jewelry/` into the content store:

```bash
uv run python scripts/migrate_to_library.py
```

This walks the input tree, registers each file via `ContentStore`, and
emits collection JSONs under `library/collections/` so the same product
sets and references can be referred to by stable IDs in future runs.
