"""Dense (fp16) OpenVINO export via optimum-cli."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

import ov_converter.settings as S


def _env() -> dict:
    env = os.environ.copy()
    env.setdefault(S.HF_HOME_ENV, str(S.CACHE_ROOT))
    env.setdefault(S.HF_HUB_CACHE_ENV, str(S.CACHE_ROOT / "hub"))
    return env


def export_dense(model_dir: str | Path, out_dir: str | Path, task: str,
                 log: Callable[[str], None] | None = None,
                 extra_args: list[str] | None = None) -> int:
    """optimum-cli export openvino --model <dir> --task <task> --weight-format fp16 <out>"""
    S.apply_env()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        S.env_script("optimum-cli.exe"), "export", "openvino",
        "--model", str(model_dir),
        "--task", task,
        "--trust-remote-code",
        "--weight-format", "fp16",
    ]
    if extra_args:
        cmd += extra_args
    cmd.append(str(out_dir))

    def emit(line: str) -> None:
        if log:
            log(line)

    emit("Running: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=_env())
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            emit(line)
    proc.wait()
    return proc.returncode


# ------------------------------------------------------------------ submodel helpers
def list_submodels(ir_dir: str | Path) -> list[str]:
    """openvino_*.xml IRs to compress (excludes tokenizer/detokenizer)."""
    d = Path(ir_dir)
    out = []
    for f in sorted(d.glob("openvino_*.xml")):
        name = f.stem
        if name in ("openvino_tokenizer", "openvino_detokenizer"):
            continue
        out.append(f.name)
    if not out and (d / "openvino_model.xml").exists():
        out = ["openvino_model.xml"]
    return out


def submodel_stem_for(mode: str, original_stem: str) -> str:
    """e.g. openvino_language_model -> openvino_language_model (name unchanged)."""
    return original_stem
