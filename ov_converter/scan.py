"""Local model scanner: find dense convertible models, exclude quantized/GGUF/OV/cache."""
from __future__ import annotations

from pathlib import Path

import ov_converter.settings as S
from ov_converter.hf import task_from_config, read_config

EXCLUDE_DIRS = {".cache", ".git", "logs", ".locks", "hub", "xet"}
SKIP_PREFIXES = (".", "models--")


def _excluded(name: str) -> bool:
    return name in EXCLUDE_DIRS or name.startswith(SKIP_PREFIXES)


def _config_size(d: Path) -> int:
    s = 0
    for pat in ("*.safetensors", "*.bin"):
        for f in d.glob(pat):
            s += f.stat().st_size
    return s


def scan_models(root: str | Path | None = None) -> list[dict]:
    r"""Suitable source models under T:\models\** (org/model dirs)."""
    root = Path(root) if root else S.ORIGINALS_ROOT
    out: list[dict] = []
    if not root.exists():
        return out

    # depth-2 walk: T:\models\<org>\<model>
    for org in sorted(root.iterdir()):
        if not org.is_dir() or _excluded(org.name):
            continue
        for d in sorted(org.iterdir()):
            if not d.is_dir() or _excluded(d.name):
                continue
            info = _classify(d)
            if info:
                out.append(info)
    out.sort(key=lambda x: x["path"].lower())
    return out


def _classify(d: Path) -> dict | None:
    """Return a record if `d` is a convertible dense model, else None."""
    cfg_path = d / "config.json"
    cfg = read_config(d) if cfg_path.exists() else {}

    is_ov = (d / "openvino_model.xml").exists() or (d / "openvino_language_model.xml").exists()
    is_gguf = bool(list(d.glob("*.gguf")))
    is_quantized = bool(cfg.get("quantization_config")) or (d / "quantization_config.json").exists()
    has_weights = (d / "model.safetensors.index.json").exists() or \
        bool(list(d.glob("*.safetensors"))) or bool(list(d.glob("pytorch_model*.bin")))

    if is_ov:
        return None                      # already converted -> separate "Converted" list
    if is_gguf:
        return None                      # GGUF source, not convertible by this flow
    if is_quantized:
        return None                      # already quantized source (e.g. AutoRound)
    if not cfg_path.exists() or not has_weights:
        return None

    size = _config_size(d)
    arch = (cfg.get("architectures") or [None])[0]
    return {
        "path": str(d),
        "name": d.name,
        "org": d.parent.name,
        "model_type": cfg.get("model_type"),
        "architectures": arch,
        "task": task_from_config(cfg),
        "size_bytes": size,
        "size_gb": round(size / 1e9, 2),
        "has_tokenizer": (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists(),
        "is_vlm": bool(cfg.get("vision_config")),
        "is_moe": _is_moe(cfg, arch),
    }


def _is_moe(cfg: dict, arch: str | None) -> bool:
    def _has_expert_key(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        return any("expert" in str(k).lower() for k in d) or \
            any(_has_expert_key(v) for v in d.values() if isinstance(v, dict))
    if _has_expert_key(cfg.get("text_config", {})) or _has_expert_key(cfg):
        return True
    return "moe" in str(arch).lower() or "moe" in str(cfg.get("model_type", "")).lower()


def scan_converted(root: str | Path | None = None) -> list[dict]:
    r"""Converted OV models in T:\models\savvadesogle (or custom root)."""
    root = Path(root) if root else S.OUTPUT_ROOT
    out: list[dict] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        is_ov = (d / "openvino_model.xml").exists() or (d / "openvino_language_model.xml").exists()
        if is_ov:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            out.append({
                "path": str(d), "name": d.name,
                "size_gb": round(size / 1e9, 2),
                "is_vlm": (d / "openvino_vision_embeddings_model.xml").exists(),
                "has_tokenizer": (d / "openvino_tokenizer.xml").exists(),
            })
    out.sort(key=lambda x: x["name"].lower())
    return out
