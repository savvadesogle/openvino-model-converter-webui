"""Versions of the libraries that matter for conversion."""
from __future__ import annotations

import importlib.metadata as im
import json
import subprocess
import sys
import time

import ov_converter.settings as S

CORE_LIBS = [
    "openvino",
    "openvino-genai",
    "openvino-tokenizers",
    "nncf",
    "optimum",
    "optimum-intel",
    "transformers",
    "torch",
    "huggingface_hub",
    "compressed-tensors",
    "fastapi",
    "uvicorn",
]

_RESOLVED_SCRIPT = "\n".join([
    "import sys, json, importlib.metadata as m;",
    f"libs = {json.dumps(CORE_LIBS)};",
    "out = {};",
    "for l in libs:",
    "    try: out[l] = m.version(l)",
    "    except Exception: out[l] = None",
    "out['python'] = sys.version.split()[0];",
    "print(json.dumps(out))",
])
_RESOLVED_TTL = 30.0
_resolved_cache: dict = {}


def _invalidate_resolved() -> None:
    _resolved_cache.clear()


def get_version(dist: str) -> str | None:
    try:
        return im.version(dist)
    except im.PackageNotFoundError:
        return None


def versions() -> dict[str, str | None]:
    out = {"python": sys.version.split()[0]}
    for lib in CORE_LIBS:
        out[lib] = get_version(lib)
    return out


def resolved_versions() -> dict[str, str | None]:
    now = time.time()
    if _resolved_cache and _resolved_cache.get("ts") and (_resolved_cache["ts"] + _RESOLVED_TTL) > now:
        return _resolved_cache.get("value", {})
    out: dict[str, str | None] = {}
    try:
        r = subprocess.run([S.resolve_python(), "-c", _RESOLVED_SCRIPT],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=60, cwd=str(S.PROJECT_DIR))
        if r.returncode == 0 and (r.stdout or "").strip():
            parsed = json.loads((r.stdout or "").strip())
            if isinstance(parsed, dict):
                out = parsed
    except Exception:  # noqa: BLE001
        out = {}
    _resolved_cache["ts"] = time.time()
    _resolved_cache["value"] = out
    return out


def invalidate() -> None:
    _invalidate_resolved()
