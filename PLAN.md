# Plan: `openvino-model-converter-webui`

Download Hugging Face models and convert them to OpenVINO INT2/INT3/INT4/...
with a minimal two-tab web UI.

## Global rules (hard-coded in `ov_converter/settings.py`)

| What | Path |
|---|---|
| HF + xet cache | `T:\models\.hf-cache` (env `HF_HOME`, `HF_HUB_CACHE`) |
| Original models | `T:\models\<org>\<model>` |
| Converted output | `T:\models\savvadesogle\<Base>-<mode>-ov` |
| Intermediate dense (fp16 IR) | `T:\models\savvadesogle\<Base>-fp16-ov` (checkbox "delete after") |
| Tool itself | `T:\tools\ov-converter\` |

Naming follows the HF `OpenVINO` org convention (`Qwen3-8B-int4-ov`,
`whisper-large-v3-turbo-int8-ov`): `<Base>-<mode>-ov`, `-ov` is always last.
Mixed AutoRound-style scheme token: `int2-mix` -> `<Base>-int2-mix-ov`.

## Architecture (microservices; download is separate from convert)

```
T:\tools\ov-converter\
├── requirements.txt / pyproject.toml / README.md / PLAN.md
├── ov_converter/                  # reusable core (no UI)
│   ├── settings.py                # paths, env HF_HOME -> T:, naming rules
│   ├── naming.py                  # <Base>-<mode>-ov
│   ├── modes.py                   # dynamic NNCF mode list + per-mode self-test
│   ├── versions.py                # versions of key libraries
│   ├── checks.py                  # disk / virtual-memory(pagefile) / param validation
│   ├── hf.py                      # link parse, validation, model_info, download  (DOWNLOAD)
│   ├── scan.py                    # local model scanner for the dropdown
│   ├── export.py                  # dense fp16 export via optimum-cli          (CONVERT)
│   ├── compress.py                # NNCF compress per submodel, two-pass int2mix(CONVERT)
│   ├── package.py                 # copy tokenizer/config, HF README model card, manifest.json
│   ├── genai_test.py              # E2E check in OpenVINO GenAI
│   ├── pipeline.py                # stage orchestration, emits @@events on stdout
│   └── cli.py                     # `python -m ov_converter.pipeline <config.json>`
├── webui/
│   ├── main.py                    # FastAPI app: REST + SSE, spawns pipeline subprocess
│   ├── tasks.py                   # single-task manager (lock, queue, cancel)
│   ├── logbus.py                  # per-task log pub/sub
│   └── static/ index.html, style.css, app.js
└── scripts/
```

Each leaf module is standalone & importable (`hf.download()`, `compress.compress_ir()`,
`naming.name_output()`, ...). `pipeline.py` only composes them.

## Framework

FastAPI + uvicorn (uvicorn already present; install `fastapi`). Native async SSE,
Pydantic validation, clean API/UI split. Vanilla HTML/CSS/JS (no build step).

## Download tab

- Input accepts `https://huggingface.co/Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-0.8B`,
  or a local path.
- Validation via `HfApi().model_info(...)`: exists, gated/needs token, model type,
  pipeline task, license, file count + total size.
- Options: revision, HF_TOKEN, "only what is needed for conversion" (include
  `*.safetensors`, `*.json`, `*.txt`, `*.jinja`, `merges.txt`, `vocab.json`).
- Command: `hf download <id> --local-dir T:\models\<org>\<model> ...`
  (env on T: set automatically by the app).
- Post-download check: `config.json`, weights, tokenizer files present; show size.

## Convert tab

- Model: dropdown from `scan.py` (sources in `T:\models\**`) + paste link/path,
  with "download first" checkbox.
- Task auto-detected from `config.json` (`vision_config` -> image-text-to-text,
  else text-generation).
- Mode: dynamic list from current `nncf.CompressWeightsMode` filtered by the OV
  support map + optional "Self-test modes" (tiny compress+compile per mode).
- Params: group_size (-1/32/64/128/256), all_layers, ratio + backup_precision,
  hidden "Data-aware" block (awq / scale_estimation / gptq + dataset/num_samples).
- VLM submodels: checkbox "compress only language_model + text_embeddings, keep
  vision in fp16" (default on).
- Output name preview (`T:\models\savvadesogle\Qwen3.5-0.8B-int2-ov`), editable.
- Param validation before run (group_size divisibility, int8 -> g=-1, int2mix MoE only).
- Stage timeline: Validate -> Download -> Export -> Compress -> Package ->
  Tokenizer check -> GenAI test, with spinners + animated checkmarks, per-stage logs.

## Checks that gate the buttons

- Disk: `shutil.disk_usage`; needed = model size (download) or fp16 IR + result
  (convert); button disabled until "Free X GB / Need Y GB" passes.
- RAM: Windows `GlobalMemoryStatusEx` (virtual memory incl. pagefile); peak
  estimate = params*2*2 bytes; warn/block when insufficient.

## Post-conversion

- `package.py` writes a HF-compatible **README.md model card** (same style as
  `OpenVINO/Qwen3.8-27B-int4-ov`, excluding what does not apply), a
  `manifest.json` (mode, group_size, versions, commands) and copies tokenizer,
  config, preprocessor/processor files. `genai_test.py` runs the model through
  GenAI `LLMPipeline`/`VLMPipeline` and reports output + tok/s.

## Versions panel

`openvino`, `openvino-genai`, `nncf`, `optimum`, `optimum-intel`, `transformers`,
`torch`, `huggingface_hub`, `compressed-tensors`, `python`, `fastapi/uvicorn`,
compared against `requirements.txt`.

## Design / language

Flat minimal UI, pastel colors, all terms in English, `ⓘ` help on every block.

## Repo

Name: `openvino-model-converter-webui`. GitHub access: local `gh` needs
`gh auth login -h github.com` (current token invalid); fine-grained PAT scoped to
the repo (Contents write) can be used for CI / remote agents via `GH_TOKEN`.
