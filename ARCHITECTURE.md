# Architecture: openvino-model-converter-webui

> A self-contained onboarding document. A new agent can use this file alone to
> understand the project, run it, extend it, and test changes — no other
> context is required.

## 1. What this project does

A local web application (FastAPI + vanilla JS) with two tabs:

- **Tab 1 — Download**: paste a Hugging Face link / model id (`Qwen/Qwen3.5-0.8B`)
  or a local path, validate it, and download it into a configurable directory.
- **Tab 2 — Convert**: pick a downloaded model, choose an NNCF weight-compression
  mode (`int2`, `int3`, `int4`, `int8`, `nf4`, `mxfp4`, `fp8`, `cb4`, `int2-mix`…),
  export it to OpenVINO (dense fp16 IR) and compress the weights, then package a
  HF-compatible model card + `manifest.json` and run an end-to-end GenAI sanity test.

The core pipeline is also available headless via CLI (`python -m ov_converter.pipeline config.json`).

Reference/working environment (as of this document):

```
python    3.12.13
torch     2.13.0 (CPU)
openvino  2026.4.0.dev20260814
openvino-genai  2026.4.0.0.dev20260814
openvino-tokenizers 2026.4.0.0.dev20260814
nncf      3.3.0
optimum   2.3.0
optimum-intel 2.2.0.dev0
transformers 5.2.0        <- IMPORTANT: pinned, see §7
huggingface_hub 1.21.0
compressed-tensors 0.18.0
fastapi   0.141.1
uvicorn   0.38.0
```

## 2. Repository layout

```
T:\tools\ov-converter\
├── README.md             user-facing readme (screenshot, usage)
├── ARCHITECTURE.md       this document
├── PLAN.md               original design plan
├── requirements.txt      pinned/expected versions (transformers==5.2.0!)
├── pyproject.toml        package metadata; console script ov-converter
├── .gitattributes        line-ending normalization (eol=lf)
├── .gitignore            ignores __pycache__, logs/, .env, .venv
├── LICENSE               Apache-2.0
├── docs/screenshot.png   UI screenshot used in README
├── ov_converter/         reusable core (no UI) — importable, "microservice" modules
│   ├── settings.py       paths, env (HF cache on disk), python-env resolution
│   ├── naming.py         output naming: <Base>-<mode>-ov
│   ├── modes.py          dynamic NNCF mode list + per-mode self-test
│   ├── versions.py       versions of key libraries
│   ├── checks.py         disk / virtual-memory(pagefile) / param validation
│   ├── hf.py             HF link parsing, validate, download, local_check, verify_hashes
│   ├── scan.py           local model scanner (excludes GGUF/OV/quantized)
│   ├── export.py         dense fp16 OpenVINO export via optimum-cli + submodel listing
│   ├── compress.py       NNCF compression (single mode + two-pass int2/int3-mix)
│   ├── package.py        copy tokenizer/config, HF README card, manifest.json, verify()
│   ├── genai_test.py     E2E via openvino-genai LLMPipeline / VLMPipeline
│   ├── pipeline.py       stage orchestration; emits @@events on stdout; CLI entry
│   └── __init__.py
└── webui/                FastAPI app + single-page UI
    ├── main.py           routes, SSE, static mount
    ├── tasks.py          single-task manager (subprocess + log stream + cancel)
    ├── __init__.py
    └── static/
        ├── index.html    two tabs, cards, task panel
        ├── style.css     flat pastel, square blocks, responsive
        └── app.js        all frontend logic (state, fetch, SSE, stage chips)
```

`logs/` holds task config JSON + uvicorn logs (gitignored).

## 3. Paths, naming, environment

All paths come from `ov_converter/settings.py` (cross-platform via `pathlib.Path`):

| Symbol | Windows default | Linux default | Purpose |
|---|---|---|---|
| `MODELS_ROOT` | `T:\models` | `~/models` (env `OV_MODELS_ROOT` overrides) | root for originals + output |
| `ORIGINALS_ROOT` | `= MODELS_ROOT` | same | downloaded `T:\models\<org>\<model>` |
| `OUTPUT_ROOT` | `MODELS_ROOT\savvadesogle` | same | converted models |
| `CACHE_ROOT` | `MODELS_ROOT\.hf-cache` | same | HF_HOME + xet cache |
| `PROJECT_DIR` | derived from `__file__` | same | repo root |
| intermediate fp16 IR | `<OUTPUT_ROOT>\<Base>-fp16-ov` | same | dense export before compression |

Naming (HF OpenVINO convention): `<Base>-<mode>-ov`, `-ov` always last.
Mode tokens: `int2, int3, int4, int8, nf4, mxfp4, mxfp8, fp8e4m3, cb4, int2-mix, int3-mix, fp16`.
Intermediate dense: `<Base>-fp16-ov` (deleted after conversion by default).

`settings.apply_env()` sets `HF_HOME`/`HF_HUB_CACHE` (setdefault) so everything on this
machine caches to `T:`; call it at startup (webui.main already does).

`settings.resolve_python()` picks the interpreter that can run the whole pipeline
(openvino + nncf + optimum + transformers 5.2). It prefers `sys.executable`, else scans
conda envs. `settings.env_script(name)` resolves CLI scripts (e.g. `optimum-cli.exe`,
`hf.exe`) inside that env's `Scripts` dir so subprocesses never pick up the wrong conda env.

## 4. Core modules (microservice style — each importable on its own)

- **hf.py**
  - `parse_hf_id(text) -> str|None`: accepts `org/model`, `org\model`, `huggingface.co/...`,
    `https://huggingface.co/.../tree/main`, quotes/spaces → normalized `org/model`.
  - `validate_model_id(id, token) -> dict`: `model_info(files_metadata=True)` → `{ok, id, sha,
    pipeline_tag, tags, gated, license, files, total_bytes, total_gb, files_meta:[{name,size,sha256}],
    local_* presence fields}`.
  - `detect_local(path) -> dict`: local dir → same rich shape (files_meta, total_gb, license,
    `id` from `_name_or_path` if present).
  - `download(id, dest, *, revision, token, include_only, log, progress) -> int`: per-file
    `hf_hub_download` loop (progress callback), deletes `dest/.cache` afterwards; token falls
    back to `HF_TOKEN` env; returns 0/1.
  - `local_check(path, files) -> dict`: `{exists, complete, missing/missing_files,
    present_files, present, total, size_gb}`. Skips files inside dot-DIRECTORIES (`.cache`,
    `.git`) but keeps dot-FILES (`.gitattributes`).
  - `verify_hashes` / `verify_hashes_stream`: sha256 for LFS files, SIZE fallback for non-LFS
    (`method: "sha256"|"size"`), generator events `start/file/done`.
- **scan.py** — walks `T:\models\<org>\<model>`; excludes `.cache`, `models--*`, `.*`,
  GGUF (has `*.gguf`), already-OV (has `openvino_model.xml`), quantized (config has
  `quantization_config`). Each record: name, model_type, task (from config: vision→
  `image-text-to-text`, audio→ASR, else `text-generation`), size, is_vlm, is_moe.
- **export.py** — `export_dense(model_dir, out_dir, task)` runs
  `optimum-cli export openvino --weight-format fp16` via `env_script`; `list_submodels()`
  returns `openvino_*.xml` except tokenizer/detokenizer.
- **compress.py** — `compress_ir(ir, out, mode, group_size, ...)`, `compress_dir(src,dst,...)`
  (compresses text submodels, copies vision fp16 when `only_text`), two-pass `int2_mix`
  (experts `.*mlp\.experts\..*` → int2, rest → int4). Uses `nncf.compress_weights`.
- **package.py** — `copy_metadata`, `verify` (tokenizer/config present),
  `write_manifest`, `generate_readme` (HF model card like `OpenVINO/*-int4-ov`).
- **genai_test.py** — `run_test(model_dir, ...)` loads `LLMPipeline`/`VLMPipeline` (VLM uses
  `ov.Tensor(_sample_image())` — images must be OV tensors, not paths), returns
  `{ok, is_vlm, prompt, output, tokens, elapsed_s, tok_per_s}`.
- **checks.py** — `disk_free/disk_check`, `virtual_memory()` (Windows `GlobalMemoryStatusEx`
  incl. pagefile), `ram_check`, params estimates, `validate_convert(mode, group_size, ...)`.
- **modes.py** — `list_modes()` from `nncf.CompressWeightsMode` + curated OV map;
  `self_test_all()` runs a tiny compress+compile per mode.
- **naming.py** — `output_name(base, mode)`, `intermediate_name`, `output_dir/intermediate_dir`.
- **versions.py** — `versions()` dict via `importlib.metadata`.

## 5. Pipeline (ov_converter/pipeline.py) — stages & @@events

`ConvertConfig` dataclass fields: `model_id, model_path, dest, download, revision, token,
task, mode, group_size, all_layers, ratio, backup, data_aware, only_text,
delete_intermediate, output_dir, intermediate_dir, run_genai_test, prompt,
keep_fp16_export, download_only, include_only, files, hf_home, hf_hub_cache`.

Stages (each emits `@@STAGE <stage> | start|done|fail <detail>`; free text as
`@@LOG <stage> | text`; progress as `@@PROGRESS <stage> <pct>`; final result as
`@@META done | <json>`):

```
validate → download (if needed) → export (dense fp16) → compress → package →
tokenizer check → genai_test  (download_only stops after download)
```

- Validate: resolve source (local path / already-downloaded dir), check mode availability
  and param rules; emit `done source=<path>` or just `ok`.
- Download: `hf.download` with `@@PROGRESS`; `.cache` removed.
- Export: `optimum-cli` fp16 → `<Base>-fp16-ov`; submodel list logged.
- Compress: `compress_dir` per submodel; report dict in `done`.
- Package: copy metadata, write `manifest.json`, generate `README.md` model card.
- Tokenizer: `package.verify` — all 4 tokenizer files present.
- GenAI test: `run_test` on the converted dir; logs output + tok/s.
- Cleanup: intermediate fp16 dir removed if `delete_intermediate`.

CLI: `python -m ov_converter.pipeline <config.json>` (cwd = repo root so `ov_converter` is
importable). Emits `@@` lines to stdout for the web UI to consume.

## 6. Web UI

### Backend (webui/main.py)
`S.apply_env(); S.ensure_dirs()` on import. Serves `static/index.html` at `/`.

API routes:

| Route | Method | Purpose |
|---|---|---|
| `/api/info` | GET | paths (cache/originals/output/project), disk_free_gb, virtual_memory, versions |
| `/api/drives` | GET | available drive roots: `["T:\\","C:\\",...]` on Windows, `["/"]` elsewhere |
| `/api/modes` | GET | dynamic mode list |
| `/api/modes/self-test` | POST | run per-mode compress+compile on a tiny model |
| `/api/models` | GET | `{sources, converted}` from `scan` |
| `/api/model/estimate` | POST `{path}` | size_gb + params |
| `/api/hf/validate` | POST `{text, token}` | parse → local `detect_local` or HF `validate_model_id`; `{kind, info}` |
| `/api/model/local-check` | POST `{path, files}` | `local_check` (per-file presence) |
| `/api/model/verify-hash` | POST `{path, files}` | NDJSON stream: `start/file/done` (sha256/size) |
| `/api/disk` | GET `?path=` | `shutil.disk_usage` → free_gb |
| `/api/open-dir` | POST `{path}` | `os.startfile` (opens Explorer) |
| `/api/mkdir` | POST `{path}` | create dirs, **guarded to `MODELS_ROOT` only** (refuses other drives) |
| `/api/download` | POST | start a download-only task (one at a time) |
| `/api/convert` | POST | start a convert task |
| `/api/task/cancel` | POST | cancel the running task |
| `/api/task/status` | GET | `{busy, task:{...}}` |
| `/api/task/stream` | GET (SSE) | live log lines of the current task; ends with `done` event |

`/api/download` and `/api/convert` bodies carry `hf_home`/`hf_hub_cache` (user-edited cache
env), `dest`, `files` (selected files), etc.

### Task manager (webui/tasks.py)
One global task at a time. `Task.start()` writes a **redacted** config (token removed) to
`logs/task_<id>.json`, spawns `resolve_python() -m ov_converter.pipeline <cfg>` with env
`HF_HOME`/`HF_HUB_CACHE` (overridden from config if provided) and `HF_TOKEN`, reads stdout
line-by-line into a thread-safe list. `cancel()` terminates the process. SSE reads lines
since an index.

### Frontend (static/index.html + app.js + style.css)
Flat pastel, square blocks (`--radius:0`), all labels English, `ⓘ`-free (explanations are
visible `.desc`/`.hint` paragraphs). Two tabs: Download, Convert.

Key elements (ids): `dl-text, dl-validate, dl-tags, dl-info, dl-rev, dl-token,
dl-root, dl-drive, dl-sub, dl-dir-link, .dir-label, dl-local, dl-files, dl-all,
dl-none, dl-disk, dl-run, dl-spinner, dl-cancel, dl-progress, dl-hash-card,
dl-verify, dl-vspinner, dl-hashres, hf-drive, hf-base, hf-derived, hf-env-res,
cv-model, cv-task, cv-mode, cv-group, cv-backup, cv-ratio, cv-all-layers, cv-awq,
cv-scale, cv-gptq, cv-lora, cv-dataset, cv-nsamples, cv-only-text, cv-outdir,
cv-delete-int, cv-genai, cv-disk, cv-ram, cv-errors, cv-run, cv-spinner,
cv-cancel, cv-progress, task-panel, task-status, task-collapse, task-stages/stages,
task-log, task-cancel(STOP), task-clear, panel-download, panel-convert, dl-path-spinner
(JS-created), hash-pct/hash-list (JS-created), create-btn-* (JS-created)`.

`app.js` state object holds: `info, modes, sources, converted, dlInfo, cvInfo,
currentMode, currentTask, taskActive, taskKind, dlFiles, dlSelected (Set), dlNeededBytes,
dlLocalComplete, dlLocalExists, dlSubDirty, validatedSub, drives, validating`.

Notable behavior:
- `osSep()` returns `\\` on Windows, `/` elsewhere; drive lives only in `#dl-drive`/`#hf-drive`
  (dropdowns), path inputs hold the path without the drive.
- Drive sync: `loadDrives()`/`syncDriveFromRoot`/`syncRootFromDrive` (Destination) and
  `loadHfDrives`/`syncHfDriveFromBase`/`syncHfBaseFromDrive` (Cache env); defaults to the
  drive of `paths.originals`.
- `composeDest()` = `#dl-drive` + sep + `#dl-root` + sep + `normalizeSub(#dl-sub)`.
- Validate: `validateDownload()` locked with `state.validating` (button disabled, global
  `[data-fill]` no-ops) to avoid HF rate-limit; clears tags/info/files before each run.
- Model-overwrite guard: after a successful HF validate, `validatedSub` is set; editing
  `#dl-sub` to a different `org/model` sets `dlSubDirty` + warning banner and disables
  Download until re-validate.
- Local check: `updateLocalStatus()` calls `/api/model/local-check`, renders `#dl-local`
  banner (ok/bad/warn) and toggles `#dl-dest-card` `done` (green) / `pending` (yellow).
- File picker: per-file checkboxes; present (downloaded) files are checked+disabled;
  `#dl-all`/`#dl-none` select all/none; Download needs ≥1 selected file.
- Hash verify: `verifyHashes()` streams NDJSON; `#hash-pct` shows `Verifying hashes… NN%`,
  `#hash-list` rows update per file (`✓ ok (ref sha256: …)` / `✓ ok (size …)` /
  `✕ MISMATCH` / `✕ size mismatch` / `no reference (not LFS)` / `not present`).
- Task panel: stage chips always show the full list `validate download export compress
  package tokenizer genai_test`; `setStage(name,status)` toggles running/done/fail;
  done log uses past-tense labels (`STAGE_DONE_LABEL`); `#task-collapse` toggles collapsed
  (collapsed hides only `#task-log` + `.row`); `#task-cancel` labelled `STOP`; status chip
  shows `<kind> completed` / `<kind> FAILED` (`.chip.failed` red).

### SSE event format (emitted by pipeline, parsed by app.js)
```
@@STAGE <stage> | start|done <detail>|fail <detail>
@@LOG <stage> | <free text>
@@PROGRESS <stage> <pct>
@@META done | <json>
```
Anything not starting with `@@` is treated as a raw log line.

## 7. Critical environment gotchas

1. **transformers is pinned to `==5.2.0`** (requirements.txt). optimum 2.3.0 exports
   `qwen3_5`/`qwen3_5_moe` ONLY with transformers 5.2.x
   (`Qwen3_5OpenVINOConfig MIN=5.2.0 MAX=5.2.99`, needs `Qwen3_5DynamicCache`). Newer
   transformers breaks the OpenVINO exporter import.
2. **Subprocess env**: always launch pipeline/CLI with `settings.resolve_python()`
   (an interpreter that has openvino+nncf+optimum+transformers 5.2), and CLI tools via
   `settings.env_script(...)` — otherwise the wrong conda env gets picked up from PATH.
3. **HF cache on disk**: `apply_env()` points `HF_HOME`/`HF_HUB_CACHE` to `MODELS_ROOT/.hf-cache`
   (all on `T:`). User can override per-task via `hf_home`/`hf_hub_cache`.
4. **`/api/mkdir` is guarded** to `MODELS_ROOT` only (refuses other drives / network shares).
5. **GenAI VLM test**: pass images as `ov.Tensor` (numpy arrays), NOT file paths; console must
   use utf-8 (set `PYTHONIOENCODING=utf-8` when printing output).
6. **Do not use `hf download --include`** (new CLI mis-handles it); use per-file
   `hf_hub_download` (the project's `hf.download`).
7. **INT8 needs per-channel (`group_size=-1`)**; INT2/INT3 are symmetric-only;
   MXFP4/MXFP8 fixed group 32; `int2-mix`/`int3-mix` require a MoE model.

## 8. How to run

```powershell
# from the repo root (T:\tools\ov-converter) in the "openvino-latest" env:
pip install -r requirements.txt
uvicorn webui.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

Headless:
```powershell
python -m ov_converter.pipeline logs\some_config.json
```
(A config JSON is a `ConvertConfig` dict; see §5.)

## 9. Testing / QA workflow

- Frontend syntax: `node --check webui/static/app.js`.
- Backend syntax: `python -m py_compile ov_converter/*.py webui/*.py`.
- API smoke: FastAPI `TestClient` against `webui.main.app` (see §6 routes). The model
  `T:\models\Qwen\Qwen3.5-0.8B` is a known-good fixture (13 files, all present).
- Cross-file consistency: every `$("...")` id in `app.js` must exist in `index.html`
  exactly once (or be JS-created: `dl-path-spinner`, `hash-pct`, `hash-list`, `hash-<i>`,
  `create-btn-*`).
- Suggested per-change checks: run the pipeline on the small `Qwen3.5-0.8B` (download →
  export fp16 → int2 → package → GenAI test) end to end.

## 10. Commit conventions

Detailed messages: `<type>(<scope>): summary` + body bullets describing what/why/files.
`type` in `feat|fix|ui|chore|refactor`. Use `git commit -F <msgfile>` to avoid shell
quoting issues with Unicode (`✓`, `▾`, `↗`). Push with `git push` (creds from `gh`/keyring).

## 11. Known extension points

- Add a mode: extend `modes.py` `OV_SUPPORTED` + `naming.MODE_TOKENS` + `compress.MODE_ENUM`
  (+ self-test auto-covers it).
- Add a pipeline stage: new `emit.start("<stage>")` block in `pipeline.py::run` + add the
  stage to `STAGES` in `app.js` and `STAGE_DONE_LABEL`.
- Add an API route: FastAPI handler in `webui/main.py`; wire a button in `index.html` +
  `app.js`.
