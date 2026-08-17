"""Transformers version requirements for OpenVINO export, derived from the installed optimum exporter."""
from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

import ov_converter.settings as S

TF_CONSTRAINTS = {
    "arctic": ("4.51.0", "4.53.3"),
    "aquila": ("4.57", "4.57.6"),
    "baichuan": ("4.57", "4.57.6"),
    "bitnet": ("4.57", "4.57.6"),
    "chatglm": ("4.51.0", "4.55.4"),
    "data2vec-text": ("4.57", "4.57.6"),
    "dbrx": ("4.57", "4.57.6"),
    "deci": ("4.57", "4.57.6"),
    "deepseek": ("4.51.0", "4.53.3"),
    "deepseek_ocr2": ("5.11", None),
    "deepseek_v2": ("4.51.0", "4.53.3"),
    "deepseek_v3": ("4.51.0", "4.53.3"),
    "exaone": ("4.57", "4.57.6"),
    "exaone4": ("4.57", "4.57.6"),
    "falcon_mamba": ("4.57", "5.3.0"),
    "flaubert": ("4.57", "4.57.6"),
    "fun_asr": ("4.57.0", "4.57.6"),
    "gemma": ("4.57", "5.0"),
    "gemma3n": ("5.0", None),
    "gemma3n_text": ("5.0", None),
    "gemma4": ("5.5", None),
    "gemma4_text": ("5.5", None),
    "gemma4_unified": ("5.10", "5.10.99"),
    "gemma4_unified_text": ("5.10", "5.10.99"),
    "glm": ("4.57", "5.0"),
    "got_ocr2": ("4.57", "4.57.6"),
    "granitemoehybrid": ("5.5.0", None),
    "idefics3": ("4.57", "4.57.6"),
    "internlm": ("4.57", "4.57.6"),
    "internlm2": ("4.57", "4.57.6"),
    "internvl_chat": ("4.57", "4.57.6"),
    "jais": ("4.57", "4.57.6"),
    "lfm2": ("4.57", "5.4.0"),
    "lfm2_moe": ("5.0", "5.4.0"),
    "llama4": ("4.57", "4.57.6"),
    "llama4_text": ("4.57", "4.57.6"),
    "llava-qwen2": ("4.51.0", "4.53.3"),
    "llava_next_video": ("4.57", "4.57.6"),
    "mamba": ("4.57", "5.3.0"),
    "marian": ("4.57", "4.57.6"),
    "minicpm": ("4.51.0", "4.53.3"),
    "minicpm3": ("4.51.0", "4.53.3"),
    "minicpmo": ("4.51.0", "4.51.3"),
    "minicpmv": ("4.57", "4.57.6"),
    "mt5": ("4.57", "4.57.6"),
    "muse_glimmer": ("5.15.0", None),
    "muse_glimmer_text": ("5.15.0", None),
    "nystromformer": (None, "4.50.3"),
    "orion": ("4.57", "4.57.6"),
    "ouro": ("4.53.0", "4.57.6"),
    "phi3_v": ("4.51.0", "4.53.3"),
    "phi4_multimodal": ("4.51.0", "4.53.3"),
    "phi4mm": ("4.51.0", "4.53.3"),
    "qwen": ("4.51.0", "4.55.4"),
    "qwen2_5_vl": ("4.57", "5.0"),
    "qwen2_vl": ("4.57", "5.0"),
    "qwen3_5": ("5.2.0", "5.2.99"),
    "qwen3_5_moe": ("5.2.0", "5.2.99"),
    "qwen3_5_moe_text": ("5.2.0", "5.2.99"),
    "qwen3_5_text": ("5.2.0", "5.2.99"),
    "qwen3_asr": ("4.57.6", "4.57.6"),
    "qwen3_next": ("4.57", "4.57.6"),
    "qwen3_omni_moe": ("5.0", None),
    "qwen3_omni_moe_talker_text": ("5.0", None),
    "qwen3_omni_moe_text": ("5.0", None),
    "qwen3_vl": ("4.57", "5.0"),
    "smolvlm": ("4.57", "4.57.6"),
    "videochat_flash_qwen": ("4.57", "4.57.6"),
    "xlm": ("4.57", "4.57.6"),
    "xverse": ("4.57", "4.57.6"),
    "zamba2": ("4.57", "4.57.6"),
}


def _norm(v: str) -> str:
    if not v:
        return ""
    v = v.split("+", 1)[0]
    m = re.match(r"(\d+(?:\.\d+)*)", v.strip())
    if not m:
        return ""
    parts = (m.group(1).split(".") + ["0", "0", "0"])[:3]
    return ".".join(parts)


def _ver_tuple(v: str) -> tuple[int, int, int]:
    parts = _norm(v).split(".")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return (0, 0, 0)


def _concrete(v: str | None) -> bool:
    return bool(v and re.fullmatch(r"\d+\.\d+\.\d+", v))


def _satisfies(ver: tuple[int, int, int],
               lo: tuple[int, int, int] | None,
               hi: tuple[int, int, int] | None) -> bool:
    if lo is not None and ver < lo:
        return False
    if hi is not None and ver > hi:
        return False
    return True


def required_transformers(cfg: dict, installed: str | None = None) -> dict:
    cfg = cfg if isinstance(cfg, dict) else {}
    model_type = cfg.get("model_type")
    raw_floor = cfg.get("transformers_version")
    floor = _norm(raw_floor) if raw_floor else None
    if installed is None:
        installed = installed_version()
    min_v, max_v = TF_CONSTRAINTS.get(model_type, ("4.57", None))
    if model_type is None:
        return {
            "model_type": None, "floor": floor,
            "constraint_min": min_v, "constraint_max": max_v,
            "required": "?", "recommended": None,
            "installed": installed, "ok": False, "mode": "unknown",
            "reason": "unknown model type — cannot determine required transformers version",
        }
    lo = _ver_tuple(min_v) if min_v else None
    hi = _ver_tuple(max_v) if max_v else None
    if min_v and max_v and _norm(min_v) == _norm(max_v):
        mode = "exact"
        required = _norm(min_v)
    elif min_v and max_v:
        mode = "range"
        m = re.match(r"^(\d+)\.(\d+)\.99$", _norm(max_v))
        if m:
            required = f"{m.group(1)}.{m.group(2)}.x"
        else:
            required = f"{min_v}..{max_v}"
    elif min_v:
        mode = "min"
        required = f">={min_v}"
    else:
        mode = "unknown"
        required = "?"
    recommended = None
    if floor and _satisfies(_ver_tuple(floor), lo, hi):
        recommended = floor
    elif min_v and _concrete(min_v):
        recommended = min_v
    ok = (installed is not None and mode != "unknown"
          and _satisfies(_ver_tuple(installed), lo, hi)
          and (floor is None or _ver_tuple(installed) >= _ver_tuple(floor)))
    if ok:
        reason = f"model {model_type} requires transformers {required}, installed {installed} — OK"
    else:
        reason = f"model {model_type} requires transformers {required}, installed {installed or 'not installed'} — export will fail"
    return {
        "model_type": model_type, "floor": floor,
        "constraint_min": min_v, "constraint_max": max_v,
        "required": required, "recommended": recommended,
        "installed": installed, "ok": ok, "mode": mode, "reason": reason,
    }


def from_config_file(path: str) -> dict:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    return required_transformers(cfg)


def installed_version() -> str | None:
    try:
        return importlib.metadata.version("transformers")
    except Exception:  # noqa: BLE001
        return None


def _run_pip(cmd: list[str], timeout: int, log=None) -> dict:
    if log:
        log(" ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, cwd=str(S.PROJECT_DIR))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "version": installed_version(),
                "output": "", "error": repr(e)[:300]}
    output = ((r.stdout or "") + (r.stderr or "")).strip()
    return {"ok": r.returncode == 0, "version": installed_version(),
            "output": output[-500:],
            "error": None if r.returncode == 0 else output[-500:] or "pip failed"}


def install_version(version: str, log=None) -> dict:
    return _run_pip([S.resolve_python(), "-m", "pip", "install", f"transformers=={version}"],
                    timeout=600, log=log)


def restore(log=None) -> dict:
    return _run_pip([S.resolve_python(), "-m", "pip", "install", "-r",
                     str(S.PROJECT_DIR / "requirements.txt")], timeout=900, log=log)