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
│   ├── support.py        runtime OV architecture-support registry (background subprocess + disk cache)
│   ├── _support_probe.py subprocess entry that builds the support registry + disk cache
│   ├── naming.py         output naming: <Base>-<mode>-ov
│   ├── modes.py          dynamic NNCF mode list + per-mode self-test
│   ├── versions.py       versions of key libraries
│   ├── checks.py         disk / virtual-memory(pagefile) / param validation
│   ├── resources.py      pre-download resource feasibility (disk/RAM per stage)
│   ├── tfreq.py          per-model transformers version requirements (switch/restore)
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

`logs/` holds task config JSON, uvicorn logs and the `ov_support.json` support-registry
cache (all gitignored).

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
    local_* presence fields, model_type, architectures, task, support}`.
  - `detect_local(path) -> dict`: local dir → same rich shape (files_meta, total_gb, license,
    `id` from `_name_or_path` if present), plus `model_type`, `architectures`, `task`, `support`.
  - In both, `support` is the verdict from `support.check_support(model_type, task)` (below).
  - `download(id, dest, *, revision, token, include_only, log, progress) -> int`: per-file
    `hf_hub_download` loop (progress callback), deletes `dest/.cache` afterwards; token falls
    back to `HF_TOKEN` env; returns 0/1.
  - `local_check(path, files) -> dict`: `{exists, complete, missing/missing_files,
    present_files, present, total, size_gb}`. Skips files inside dot-DIRECTORIES (`.cache`,
    `.git`) but keeps dot-FILES (`.gitattributes`).
  - `verify_hashes` / `verify_hashes_stream`: sha256 for LFS files, SIZE fallback for non-LFS
    (`method: "sha256"|"size"`), generator events `start/file/done`.
- **support.py** — runtime registry of `model_type`s supported by the installed OpenVINO
  exporter (from `optimum.exporters.tasks.TasksManager`, filtered to CLI-exportable types).
  Built in a BACKGROUND daemon thread at server import via `warm_start()`; the heavy import
  runs in a SUBPROCESS under `settings.resolve_python()` (`_support_probe.py`), so the web
  server's own interpreter does not need optimum. Results cached to `logs/ov_support.json`
  keyed by lib versions (optimum/optimum-intel/openvino/transformers/nncf/python) for fast
  restart. Interface: `warm_start/is_ready/get_supported/last_error/check_support`;
  `check_support(model_type, task)` returns `{ok, ready, state, model_type, task,
  supported_tasks, reason}` with `state ∈ supported | task_mismatch | unsupported | unknown`.
  `reset()` clears the in-memory registry and re-warms (used after a transformers version
  switch).
- **_support_probe.py** — subprocess entry that builds the registry + disk cache inside the
  resolved env.
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
- **checks.py** — `disk_free/disk_check`, `virtual_memory()` — cross-platform: on Windows via
  `GlobalMemoryStatusEx` (`ullAvailPageFile` is the commit limit; `avail_virtual_gb` means
  that, not address space), on Linux parses `/proc/meminfo` (`MemAvailable` + `SwapFree`) —
  `ram_check`, params estimates, `validate_convert(mode, group_size, ...)`.
- **tfreq.py** — per-model transformers version requirements for OpenVINO export, derived from
  the installed optimum exporter: a static `TF_CONSTRAINTS` table of `model_type` → min/max
  (default floor 4.57) combined with the config.json `transformers_version` floor.
  `required_transformers(cfg)` returns `{ok, required, recommended, installed, mode, reason,
  ...}` (`mode` ∈ exact | range | min | unknown); `install_version(version)` / `restore()`
  run pip via `settings.resolve_python()` (`restore` re-installs requirements.txt).
- **resources.py** — pre-download resource feasibility: `analyze(params, size_bytes,
  mode_bits, download_path, output_path, group_size, scale_bits)` returns per-stage
  (`download`/`convert`/`compress`) disk + RAM estimates vs. actual availability — disk free
  on the target drive (resolved via an ancestor walk, so not-yet-created folders work) and
  virtual memory via `checks.virtual_memory()` (the real available commit: Windows pagefile /
  Linux swap). Formulas: download = source×1.05, convert = source×1.05 + fp16_ir×1.15,
  compress = fp16_ir×1.15 + res_bytes×1.1, where compressed size =
  `params×(bits + scale_bits/group_size)/8×1.15` (scale_bits 16 for int/nf4, 8 for mxfp;
  group_size −1 → scale overhead 0). Peak RAM: export ≈ `params×4.8`, compress ≈
  `params×3.0`. Each stage dict carries `state` in `ok|warn|fail|unknown` (with `ok`,
  `need_disk_gb`/`free_disk_gb`/`need_ram_gb`/`avail_ram_gb`/`result_gb`/`estimated`/`issue`),
  the warn margin is 1.25× the need, and `overall` is the worst-state verdict; the dict also
  returns `recommendations` (human-readable fixes for failed stages). When params are unknown
  they are estimated from size (`size_bytes/2`, bf16 assumption; `estimated_params=true`).
- **modes.py** — `list_modes()` from `nncf.CompressWeightsMode` + curated OV map;
  `self_test_all()` runs a tiny compress+compile per mode.
- **naming.py** — `output_name(base, mode)`, `intermediate_name`, `output_dir/intermediate_dir`.
- **versions.py** — `versions()` dict via `importlib.metadata`.

## 5. Pipeline (ov_converter/pipeline.py) — stages & @@events

`ConvertConfig` dataclass fields: `model_id, model_path, dest, download, revision, token,
task, mode, group_size, all_layers, ratio, backup, data_aware, only_text,
delete_intermediate, output_dir, intermediate_dir, run_genai_test, prompt,
tfreq_auto_install, keep_fp16_export, download_only, include_only, files, hf_home,
hf_hub_cache`.

Stages (each emits `@@STAGE <stage> | start|done|fail <detail>`; free text as
`@@LOG <stage> | text`; progress as `@@PROGRESS <stage> | <pct>`; final result as
`@@META done | <json>`):

```
validate → download (if needed) → tfreq → export (dense fp16) → compress → package →
tokenizer check → genai_test  (download_only stops after download)
```

- Validate: resolve source (local path / already-downloaded dir), check mode availability
  and param rules; emit `done source=<path>` or just `ok`.
- Download: `hf.download` with `@@PROGRESS`; `.cache` removed.
- Tfreq: validates the model's transformers requirement via `tfreq.required_transformers`
  (before export, skipped for `mode="none"`); when mismatched it auto-installs the required
  version if `tfreq_auto_install` is set (and restores the pinned one from requirements.txt
  in a `finally` block so it runs even on later stage failure), otherwise the stage fails
  with the reason.
- Export: `optimum-cli` fp16 → `<Base>-fp16-ov`; submodel list logged. This stage **always**
  runs (for every mode, including `none`). For `mode="none"` (`fp16` token) `out_dir` is set
  equal to the fp16 intermediate dir and no compression happens.
- Compress: `compress_dir` per submodel; report dict in `done`. Skipped for `mode="none"`.
  INT8 modes are coerced to `all_layers=None`/`backup_mode=None` (NNCF rejects those for
  INT8); `data_aware` is ignored for INT8.
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
| `/api/info` | GET | paths (cache/originals/output/project), disk_free_gb, virtual_memory, versions, support `{ready, count, error}` |
| `/api/drives` | GET | available drive roots: on Windows via `GetLogicalDrives()` bitmask (falls back to an `os.path.exists` loop over A–Z), `["/"]` elsewhere |
| `/api/modes` | GET | dynamic mode list |
| `/api/modes/self-test` | POST | run per-mode compress+compile on a tiny model |
| `/api/models` | GET | `{sources, converted}` from `scan` |
| `/api/model/estimate` | POST `{path}` | size_gb + params (currently **unused by the frontend**) |
| `/api/resources` | POST `{params, size_bytes, mode_bits, download_path, output_path, group_size?, scale_bits?}` | `{resources: analyze(...)}` — reusable resource estimate; `ResourcesIn` carries optional `group_size`/`scale_bits`, which the handler forwards to `analyze` for the compressed-size formula (defaults: group_size=128, scale_bits=16). Currently **unused by the frontend** (Convert tab uses local heuristics in `estimateConvertNeeded`, not this route). |
| `/api/hf/validate` | POST `{text, token, dest?}` | parse → local `detect_local` or HF `validate_model_id`; `{kind, info}`; `info.tfreq` carries the required-transformers verdict; on ok `info.resources = resources.analyze(params, size_bytes, download_path=dest)` |
| `/api/model/local-check` | POST `{path, files}` | `local_check` (per-file presence) |
| `/api/model/verify-hash` | POST `{path, files}` | NDJSON stream: `start/file/done` (sha256/size) |
| `/api/disk` | GET `?path=` | walks up to the nearest existing ancestor, then `shutil.disk_usage` → free_gb/total_gb; returns `resolved_path` |
| `/api/open-dir` | POST `{path}` | `os.startfile` (opens Explorer) |
| `/api/mkdir` | POST `{path}` | create dirs under any local drive (Windows) / any absolute path (non-Windows); refuses UNC/network paths and relative paths |
| `/api/download` | POST | start a download-only task (one at a time) |
| `/api/convert` | POST | start a convert task (`ConvertIn` body) |
| `/api/tf/switch` | POST `{version}` | pip-install `transformers=={version}` via `resolve_python()`; gated on no busy task; on success `ov_support.reset()` re-probes the support registry |
| `/api/tf/restore` | POST | re-run `pip install -r requirements.txt` (restores the pinned transformers); gated on no busy task; on success `ov_support.reset()` |
| `/api/task/cancel` | POST | cancel the running task |
| `/api/task/status` | GET | `{busy, task:{...}}` |
| `/api/task/stream` | GET (SSE) | live log lines of the current task; ends with `done` event |

`/api/download` and `/api/convert` bodies carry `hf_home`/`hf_hub_cache` (user-edited cache
env), `dest`, `files` (selected files), etc. `ConvertIn` also declares `tfreq_auto_install`
(default false), which the frontend (`app.js` `runConvert`) sends from the `#cv-tfreq-auto`
checkbox; `/api/convert` forwards it (via `body.model_dump()`) to the pipeline, where it
triggers the env-swap-and-restore in the `tfreq` stage.

### Task manager (webui/tasks.py)
One global task at a time. `Task.start()` writes a **redacted** config (token removed) to
`logs/task_<id>.json`, spawns `resolve_python() -m ov_converter.pipeline <cfg>` with env
`HF_HOME`/`HF_HUB_CACHE` (overridden from config if provided) and `HF_TOKEN`, reads stdout
line-by-line into a thread-safe list. `cancel()` terminates the process. SSE reads lines
since an index.

### Frontend (static/index.html + app.js + style.css)
Flat pastel, square blocks (`--radius:0`), all labels English, `ⓘ`-free (explanations are
visible `.desc`/`.hint` paragraphs). Two tabs: Download, Convert.

Key elements (ids): `dl-text, dl-validate, dl-tags, dl-tfreq, dl-info, dl-rev, dl-token,
dl-root, dl-drive, dl-sub, dl-dir-link, .dir-label, dl-local, dl-files, dl-all,
dl-none, dl-disk, dl-run, dl-spinner, dl-cancel, dl-progress, dl-hash-card,
dl-verify, dl-vspinner, dl-hashres, hf-drive, hf-base, hf-derived, hf-env-res,
cv-model, cv-task, cv-mode, cv-group, cv-backup, cv-ratio, cv-all-layers, cv-awq,
cv-scale, cv-gptq, cv-lora, cv-dataset, cv-nsamples, cv-only-text, cv-outdir,
cv-delete-int, cv-genai, cv-disk, cv-ram, cv-errors, cv-tfreq, cv-tfreq-auto,
cv-tf-install, cv-tf-restore, cv-run, cv-spinner,
cv-cancel, cv-progress, task-panel, task-status, task-collapse, task-stages/stages,
task-log, task-cancel(STOP), task-clear, panel-download, panel-convert, dl-path-spinner
(JS-created), hash-pct/hash-list (JS-created), create-btn-* (JS-created)`.

The MODEL card is `#dl-model-card`; the architecture-support verdict renders into `#dl-support`
via `renderSupportBadge()`: `✓ SUPPORTED (model_type)` (green `.tag.support-ok`, card turns
`.card.done`) when supported; `✕ NOT SUPPORTED (model_type)` (`.tag.support-bad`, card turns
`.card.bad`) when unsupported; an amber task-mismatch note when the arch is supported but the
detected task is not among its `supported_tasks`; `support unknown`; or
`support check unavailable (environment)` when the registry could not be built. Both card and
badge are reset on every re-validate. `validateHfEnv()` falls back to probing the drive root
built from the drive letter without a duplicated colon (`drive.replace(/:$/, "")` + separator).

The Options card (`#dl-options-card`) is color-coded by cache-env validation
(`validateHfEnv()` + `applyOptions()`): green `.card.done` when the env is OK; yellow
`.card.pending` when the base folder is missing but its drive is reachable (`Missing folder
— use "Create now?"`, with a "Create now?" button that calls `/api/mkdir`); red `.card.bad`
when the base path is empty/invalid, the drive is absent/offline, free space is insufficient
(<10 GB), the disk check errors, or the HF token in `#dl-token` does not start with `hf_`.
The `#hf-env-res` banner always shows the exact reason; typing in `#dl-token` re-runs the
combined check locally (`applyOptions`, no network).

The Resources check card (`#dl-resources-card` / `#dl-resources`, shown after a successful
validate via `renderResources`) lists three `.res-row`s — Download / Convert / Compress —
each colored by its stage state (`.state-ok` green, `.state-warn` yellow, `.state-fail`
red, `.state-unknown` grey) with a status dot, the needed vs. available disk/RAM
(`resRowText`) and the stage `issue`. `updateDlChecks()` disables the Download button
(`#dl-run`) whenever any stage state is `fail` and appends an extra
`Resources: NOT ENOUGH — see Resources check above.` line to `#dl-disk`.

The Download tab also shows a transformers-version badge in `#dl-tfreq` (below the support
badge, via `renderTfreqBadge()`): a green `✓ transformers <v> OK (needs <required>)` chip when
`tfreq.ok`; an amber `⚠ needs transformers <required> (install <recommended>)` chip on a
mismatch; a grey `transformers version unknown` chip when the requirement can't be
determined. The badge resets on every re-validate.

The Convert tab shows a `#cv-tfreq` banner for the selected model (via `renderCvTfreq()`,
using the `tfreq` verdict attached to each scanned source): green `Transformers <v>
satisfies <required> — OK`, or amber `reason — use Install to switch to <recommended>`. An
"Auto-install required transformers" checkbox (`#cv-tfreq-auto`) makes the pipeline swap the
shared env for the run and restore it afterwards; `#cv-tf-install` / `#cv-tf-restore` call
`/api/tf/switch` / `/api/tf/restore` (`setTfBusy()` disables both during a swap).
`updateCvChecks()` blocks the Run button (`#cv-run`) on a transformers mismatch unless
auto-install is enabled.

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
- Task panel: stage chips always show the full list `validate download tfreq export
  compress package tokenizer genai_test`; `setStage(name,status)` toggles running/done/fail;
  done log uses past-tense labels (`STAGE_DONE_LABEL`); `#task-collapse` toggles collapsed
  (collapsed hides only `#task-log` + `.row`); `#task-cancel` labelled `STOP`; status chip
  shows `<kind> completed` / `<kind> FAILED` (`.chip.failed` red).

### SSE event format (emitted by pipeline, parsed by app.js)
```
@@STAGE <stage> | start|done <detail>|fail <detail>
@@LOG <stage> | <free text>
@@PROGRESS <stage> | <pct>
@@META done | <json>
```
Anything not starting with `@@` is treated as a raw log line.

## 7. Critical environment gotchas

1. **Transformers requirements are per-model, not global.** The repo pins `==5.2.0`
   (requirements.txt) and optimum 2.3.0 exports `qwen3_5`/`qwen3_5_moe` ONLY with
   transformers 5.2.x (`Qwen3_5OpenVINOConfig MIN=5.2.0 MAX=5.2.99`, needs
   `Qwen3_5DynamicCache`), but other models need different versions — e.g. `qwen3_asr`
   requires exactly 4.57.6 and `muse_glimmer` ≥5.15. Newer transformers can also break the
   OpenVINO exporter import. The UI/tool therefore detects per-model requirements
   (`tfreq.py`) and can switch the shared env via pip (`/api/tf/switch` + `/api/tf/restore`,
   gated on no busy task, swap-with-restore). `settings._PY_QUERY` accepts transformers
   >=4.51 so the env still resolves after a swap; the running server keeps working during a
   swap (no module imports transformers in-process) but a restart mid-swap could see the
   swapped version.
2. **Subprocess env**: always launch pipeline/CLI with `settings.resolve_python()`
   (an interpreter that has openvino+nncf+optimum+transformers 5.2), and CLI tools via
   `settings.env_script(...)` — otherwise the wrong conda env gets picked up from PATH.
3. **HF cache on disk**: `apply_env()` points `HF_HOME`/`HF_HUB_CACHE` to `MODELS_ROOT/.hf-cache`
   (all on `T:`). User can override per-task via `hf_home`/`hf_hub_cache`.
4. **`/api/mkdir`** creates dirs under any local drive (Windows) / any absolute path
   (non-Windows); it refuses UNC/network paths and relative paths.
5. **GenAI VLM test**: pass images as `ov.Tensor` (numpy arrays), NOT file paths; console must
   use utf-8 (set `PYTHONIOENCODING=utf-8` when printing output).
6. **Do not use `hf download --include`** (new CLI mis-handles it); use per-file
   `hf_hub_download` (the project's `hf.download`).
7. **INT8 needs per-channel (`group_size=-1`)**; INT2/INT3 are symmetric-only;
   MXFP4/MXFP8 fixed group 32; `int2-mix`/`int3-mix` require a MoE model.
8. **OV architecture-support registry**: computed in the `resolve_python()` env via a
   subprocess (`_support_probe.py`), so the web server interpreter does not need `optimum`.
   If the registry cannot be built, the UI shows "support check unavailable (environment)"
   and `/api/info` `support.error` carries the reason. The cache lives at
   `logs/ov_support.json` (gitignored). Conversions still require the resolved env.

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
  `T:\models\Qwen\Qwen3.5-2B` is a known-good fixture (13 files, all present).
- Cross-file consistency: every `$("...")` id in `app.js` must exist in `index.html`
  exactly once (or be JS-created: `dl-path-spinner`, `hash-pct`, `hash-list`, `hash-<i>`,
  `create-btn-*`).
- Suggested per-change checks: run the pipeline on the small `Qwen3.5-2B` (download →
  export fp16 → int2 → package → GenAI test) end to end.
- Architecture-support check: verify via `support.check_support` under `openvino-latest`
  (e.g. `("llama", "text-generation")` → `supported`, `("qwen3_5", "text-generation")` →
  `task_mismatch`), plus `node --check webui/static/app.js`.
- Resource analysis: verify via `resources.analyze(...)` under `openvino-latest` (e.g.
  `analyze(params=1e9, size_bytes=2e9)` → `convert.need_disk_gb` 4.4,
  `compress.result_gb` 0.6), plus `node --check webui/static/app.js`.
- Transformers requirement: `tfreq.required_transformers({"model_type": "qwen3_asr"})` →
  `recommended "4.57.6"`, `ok false` (with 5.2.0 installed), and
  `tfreq.required_transformers({"model_type": "qwen3_5", "transformers_version": "4.57.0.dev0"})`
  → `required "5.2.x"`.

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
