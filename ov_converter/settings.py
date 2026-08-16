"""Paths, environment and global rules. Everything lives on T:."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(r"T:\tools\ov-converter")
MODELS_ROOT = Path(r"T:\models")
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


def env_script(name: str) -> str:
    """Absolute path to a CLI script inside the current Python env, so subprocesses
    do not accidentally pick up another conda env from PATH."""
    base = Path(sys.executable).parent
    for cand in (base / name, base / "Scripts" / name):
        if cand.exists():
            return str(cand)
    return name
