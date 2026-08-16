"""Versions of the libraries that matter for conversion."""
from __future__ import annotations

import importlib.metadata as im
import sys

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
