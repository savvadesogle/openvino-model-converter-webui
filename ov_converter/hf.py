"""HF link parsing, validation and download."""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import Callable

import ov_converter.settings as S

HF_ID_RE = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")

# files that are enough to convert a dense model
CONVERT_INCLUDE = [
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.jinja",
    "*.py",       # trust-remote-code fallback
    "*.md",       # README (kept for reference)
]


def parse_hf_id(text: str) -> str | None:
    """Accept full URL / hf URL / bare `org/model`. Returns model id or None."""
    text = (text or "").strip().strip("\"'")
    if not text:
        return None
    text = text.replace("\\", "/")
    text = re.sub(r"^https?://", "", text).strip("/")
    if text.startswith("huggingface.co/"):
        text = text[len("huggingface.co/"):]
    elif text.startswith("hf.co/"):
        text = text[len("hf.co/"):]
    # strip trailing path like /tree/main, /blob/..., /resolve/...
    text = re.split(r"/(?:tree|blob|resolve|blame|raw)/", text)[0]
    m = HF_ID_RE.match(text)
    if not m:
        return None
    return m.group(0)


def is_local_path(text: str) -> Path | None:
    p = Path(text.strip().strip('"'))
    if p.exists():
        return p
    return None


def validate_model_id(model_id: str, token: str | None = None) -> dict:
    """Check the model exists on the Hub and return metadata for the UI."""
    from huggingface_hub import HfApi

    token = (token or "").strip() or None
    api = HfApi(token=token)
    try:
        info = api.model_info(model_id, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}

    tags = info.tags or []
    gated = bool(getattr(info, "gated", None))

    license = getattr(info, "license", None)
    card_data = getattr(info, "card_data", None)
    if not license and card_data is not None:
        license = card_data.get("license") if isinstance(card_data, dict) else getattr(card_data, "license", None)
    if not license:
        for tag in tags:
            if tag.startswith("license:"):
                license = tag.split(":", 1)[1]
                break
    if not license:
        card_data_attr = getattr(info, "cardData", {})
        license = card_data_attr.get("license") if isinstance(card_data_attr, dict) else None
    sib_names = [s.rfilename for s in info.siblings or []]
    files_meta = []
    total = 0
    for s in info.siblings or []:
        size = s.size if getattr(s, "size", 0) else (s.lfs.get("size", 0) if getattr(s, "lfs", None) else 0)
        total += size
        files_meta.append({"name": s.rfilename or getattr(s, "path", None), "size": size})

    local_dir = S.model_dir(model_id)
    local_exists = local_dir.is_dir()
    local_names: list[str] = []
    if local_exists:
        for p in local_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(local_dir)
                if any(part.startswith(".") for part in rel.parts[:-1]):
                    continue
                local_names.append(rel.as_posix())
    local_missing = [n for n in sib_names if n not in set(local_names)]
    local_complete = local_exists and not local_missing
    local_size = sum((local_dir / f).stat().st_size
                     for f in local_names if (local_dir / f).is_file())

    return {
        "ok": True,
        "id": info.id,
        "sha": info.sha,
        "pipeline_tag": info.pipeline_tag,
        "tags": tags,
        "gated": gated,
        "license": license,
        "card_data": card_data,
        "files": sib_names,
        "total_bytes": total,
        "total_gb": round(total / 1e9, 2),
        "local_dir": str(local_dir),
        "local_exists": local_exists,
        "local_files": len(local_names),
        "local_missing": local_missing,
        "local_complete": local_complete,
        "local_size_gb": round(local_size / 1e9, 2),
        "files_meta": files_meta,
    }


def task_from_config(cfg: dict) -> str:
    if cfg.get("vision_config"):
        return "image-text-to-text"
    if cfg.get("audio_config"):
        return "automatic-speech-recognition"
    return "text-generation"


def read_config(model_dir: str | Path) -> dict:
    import json

    p = Path(model_dir) / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def detect_local(model_dir: str | Path) -> dict:
    """Validate a local model dir: config present, weights present, task, type."""
    from ov_converter import checks

    d = Path(model_dir)
    cfg_path = d / "config.json"
    cfg = read_config(d)
    has_weights = (d / "model.safetensors.index.json").exists() or \
        bool(list(d.glob("*.safetensors"))) or bool(list(d.glob("*.bin")))
    return {
        "ok": cfg_path.exists() and has_weights,
        "path": str(d),
        "name": d.name,
        "config": cfg_path.exists(),
        "has_weights": has_weights,
        "task": task_from_config(cfg) if cfg else None,
        "model_type": cfg.get("model_type") if cfg else None,
        "architectures": (cfg.get("architectures") or [None])[0],
        "is_quantized": bool(cfg.get("quantization_config")),
        "is_ov": (d / "openvino_model.xml").exists() or
                 (d / "openvino_language_model.xml").exists(),
        "is_gguf": bool(list(d.glob("*.gguf"))),
        "has_tokenizer": (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists(),
        "size_gb": round(checks.dir_size(d) / 1e9, 2),
        "size_bytes": checks.dir_size(d),
        "params": checks.params_from_index(d),
    }


def local_check(path: str | Path, files: list[str]) -> dict:
    d = Path(path)
    local_names = set()
    if d.is_dir():
        for p in d.rglob("*"):
            if p.is_file():
                rel = p.relative_to(d)
                if any(part.startswith(".") for part in rel.parts):
                    continue
                local_names.add(rel.as_posix())
    missing = [n for n in files if n not in local_names]
    present_files = [n for n in files if n in local_names]
    return {
        "exists": d.is_dir(),
        "complete": d.is_dir() and not missing,
        "missing": missing,
        "missing_files": missing,
        "present_files": present_files,
        "present": len(present_files),
        "total": len(files),
        "size_gb": round(sum((d / n).stat().st_size for n in files if (d / n).is_file()) / 1e9, 2),
    }


def download(model_id: str, dest: str | Path, *, revision: str | None = None,
             token: str | None = None, include_only: bool = False,
             files: list[str] | None = None,
             log: Callable[[str], None] | None = None,
             progress: Callable[[float], None] | None = None) -> int:
    """Download a HF repo in-process, file by file; 0 on success, 1 on failure."""
    from huggingface_hub import HfApi, hf_hub_download

    S.ensure_dirs()
    S.apply_env()
    # never require the token to travel through the config file: read from env as fallback
    token = (token or os.environ.get("HF_TOKEN") or "").strip() or None
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)

    def emit(line: str) -> None:
        if log:
            log(line)

    try:
        emit(f"Downloading {model_id} -> {dest}")
        if files is None:
            files = api.list_repo_files(model_id, revision=(revision or "main"))
            if include_only:
                files = [n for n in files
                         if any(fnmatch.fnmatch(n, pat) for pat in CONVERT_INCLUDE)]
        if not files:
            emit("no files selected")
            return 0
        for i, name in enumerate(files):
            hf_hub_download(repo_id=model_id, filename=name, local_dir=str(dest),
                            revision=revision or None, token=token or None)
            emit(f"Downloading {name} ...")
            if progress:
                progress((i + 1) / len(files) * 100)
        shutil.rmtree(dest / ".cache", ignore_errors=True)
        emit(f"Downloaded {len(files)} files to {dest}")
        return 0
    except Exception as e:  # noqa: BLE001
        emit(f"Download failed: {e}")
        return 1
