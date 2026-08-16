"""HF link parsing, validation and download (snapshot_download)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

import ov_converter.settings as S

HF_URL_RE = re.compile(
    r"^(?:https?://)?(?:huggingface\.co|hf\.co)?/?(?P<id>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)

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
    text = text.strip()
    if not text:
        return None
    m = HF_URL_RE.match(text)
    if not m:
        return None
    mid = m.group("id")
    # strip trailing path like /tree/main, /blob/..., /resolve/...
    mid = re.split(r"/(?:tree|blob|resolve|blame|raw)/", mid)[0]
    return mid


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
    total = 0
    for s in info.siblings or []:
        total += s.size if getattr(s, "size", 0) else (s.lfs.get("size", 0) if getattr(s, "lfs", None) else 0)

    return {
        "ok": True,
        "id": info.id,
        "sha": info.sha,
        "pipeline_tag": info.pipeline_tag,
        "tags": tags,
        "gated": gated,
        "license": getattr(info, "license", None),
        "card_data": getattr(info, "card_data", None),
        "files": len(info.siblings or []),
        "total_bytes": total,
        "total_gb": round(total / 1e9, 2),
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


def download(model_id: str, dest: str | Path, *, revision: str | None = None,
             token: str | None = None, include_only: bool = False,
             log: Callable[[str], None] | None = None) -> int:
    """Download a HF repo in-process via snapshot_download; 0 on success, 1 on failure."""
    from huggingface_hub import snapshot_download

    S.ensure_dirs()
    S.apply_env()
    # never require the token to travel through the config file: read from env as fallback
    token = (token or os.environ.get("HF_TOKEN") or "").strip() or None
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    allow_patterns = CONVERT_INCLUDE if include_only else None

    def emit(line: str) -> None:
        if log:
            log(line)

    try:
        emit(f"Downloading {model_id} -> {dest}")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(dest),
            revision=revision or None,
            token=token or None,
            allow_patterns=allow_patterns,
            max_workers=8,
        )
        files = [p for p in dest.rglob("*") if p.is_file() and ".cache" not in p.parts]
        for f in files:
            emit(f"Downloading {f.name} ...")
        emit(f"Downloaded {len(files)} files to {dest}")
        return 0
    except Exception as e:  # noqa: BLE001
        emit(f"Download failed: {e}")
        return 1
