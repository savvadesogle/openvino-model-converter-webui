"""Paths, environment and global rules. Everything lives on T:."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _default_models_root() -> Path:
    env = os.environ.get("OV_MODELS_ROOT")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(r"T:\models")
    return Path.home() / "models"


MODELS_ROOT = _default_models_root()
ORIGINALS_ROOT = MODELS_ROOT                 # <org>/<model>
OUTPUT_ROOT = MODELS_ROOT / "savvadesogle"   # <Base>-<mode>-ov
CACHE_ROOT = MODELS_ROOT / ".hf-cache"       # HF_HOME + xet
INTERMEDIATE_SUFFIX = "-fp16-ov"

HF_HOME_ENV = "HF_HOME"
HF_HUB_CACHE_ENV = "HF_HUB_CACHE"


def apply_env() -> None:
    """Point all Hugging Face / xet caches at T:. Call once at startup."""
    os.environ.setdefault(HF_HOME_ENV, str(CACHE_ROOT))
    os.environ.setdefault(HF_HUB_CACHE_ENV, str(CACHE_ROOT / "hub"))
    # xet-py caches under $HF_HOME/xet automatically when HF_HOME is set.


def ensure_dirs() -> None:
    for d in (CACHE_ROOT, CACHE_ROOT / "hub", OUTPUT_ROOT, PROJECT_DIR / "logs"):
        d.mkdir(parents=True, exist_ok=True)


def originals_dir(org: str) -> Path:
    return ORIGINALS_ROOT / org


def model_dir(model_id: str) -> Path:
    org, _, name = model_id.partition("/")
    return originals_dir(org or "models") / name


_PY_QUERY = (
    "import importlib.util as u, importlib.metadata as m;"
    "import sys;"
    "ok = all(u.find_spec(x) is not None for x in ('openvino', 'nncf', 'optimum', 'transformers'));"
    "ok = ok and m.version('transformers').startswith('5.2');"
    "sys.exit(0 if ok else 1)"
)

_resolved_python: str | None = None


def _check_python(exe: Path) -> bool:
    try:
        r = subprocess.run([str(exe), "-c", _PY_QUERY], capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def resolve_python() -> str:
    """Pick a Python that can run the whole pipeline (openvino+nncf+optimum+transformers 5.2).

    Prefer the current interpreter; otherwise scan conda envs under the same root.
    """
    global _resolved_python
    if _resolved_python:
        return _resolved_python
    if _check_python(Path(sys.executable)):
        _resolved_python = sys.executable
        return _resolved_python
    root = Path(sys.prefix)
    while root.name not in ("miniconda3", "conda") and root.parent != root:
        root = root.parent
    candidates: list[Path] = [root / "python.exe"]
    if (root / "envs").is_dir():
        candidates += sorted((root / "envs").glob("*/python.exe"))
    for exe in candidates:
        if exe.exists() and _check_python(exe):
            _resolved_python = str(exe)
            return _resolved_python
    _resolved_python = sys.executable
    return _resolved_python


def env_script(name: str) -> str:
    """Absolute path to a CLI script inside the resolved Python env, so subprocesses
    do not accidentally pick up another conda env from PATH."""
    base = Path(resolve_python()).parent
    for cand in (base / name, base / "Scripts" / name):
        if cand.exists():
            return str(cand)
    return name
