"""Model architecture support lookup for the OpenVINO exporter, computed in the resolved env via a subprocess, with a disk cache."""
from __future__ import annotations

import json
import subprocess
import threading
import time

import ov_converter.settings as S

_ERROR: str | None = None
_SUPPORTED: dict | None = None
_ready = threading.Event()
_started = False
_lock = threading.Lock()


def _load() -> dict | None:
    global _ERROR
    script = (
        "import sys; sys.path.insert(0, r'%s'); "
        "import ov_converter._support_probe as p; p.main()" % S.PROJECT_DIR
    )
    try:
        r = subprocess.run([S.resolve_python(), "-c", script],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=180, cwd=str(S.PROJECT_DIR))
    except Exception as e:  # noqa: BLE001
        _ERROR = "subprocess failed: %s" % repr(e)[:200]
        return None
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        tail = (r.stderr or "").strip().splitlines()[-1] if (r.stderr or "").strip() else ""
        _ERROR = "probe failed (exit %s): %s" % (r.returncode, (tail or "no output")[:200])
        return None
    try:
        data = json.loads(lines[-1])
    except Exception as e:  # noqa: BLE001
        _ERROR = "invalid probe output: %s" % repr(e)[:200]
        return None
    if "error" in data:
        _ERROR = "registry build error: %s" % str(data["error"])[:200]
        return None
    _ERROR = None
    return data.get("model_types")


def warm_start() -> None:
    global _started, _SUPPORTED
    with _lock:
        if _started or _ready.is_set():
            return
        _started = True

    def _run() -> None:
        global _SUPPORTED
        _SUPPORTED = _load()
        _ready.set()

    t = threading.Thread(target=_run, name="ov-support-loader", daemon=True)
    t.start()


def is_ready() -> bool:
    return _ready.is_set()


def get_supported() -> dict | None:
    return _SUPPORTED


def last_error() -> str | None:
    return _ERROR


def check_support(model_type: str | None, task: str | None) -> dict:
    if not _ready.is_set():
        warm_start()
        deadline = time.time() + 60
        while not _ready.is_set() and time.time() < deadline:
            time.sleep(0.2)
    if not _ready.is_set():
        return {
            "ok": False, "ready": False, "state": "unknown",
            "model_type": model_type, "task": task,
            "supported_tasks": None, "reason": "support registry not ready",
        }
    registry = _SUPPORTED
    if registry is None:
        return {
            "ok": False, "ready": True, "state": "unknown",
            "model_type": model_type, "task": task,
            "supported_tasks": None,
            "reason": _ERROR or "OpenVINO exporter registry unavailable in this environment",
        }
    if not model_type:
        return {
            "ok": True, "ready": True, "state": "unknown",
            "model_type": model_type, "task": task,
            "supported_tasks": None, "reason": "no model_type in config",
        }
    entry = registry.get(model_type)
    if entry is None:
        return {
            "ok": True, "ready": True, "state": "unsupported",
            "model_type": model_type, "task": task,
            "supported_tasks": None,
            "reason": "architecture not supported by the installed OpenVINO exporter",
        }
    tasks = entry["tasks"]
    if not task or task in tasks or (task + "-with-past") in tasks:
        return {
            "ok": True, "ready": True, "state": "supported",
            "model_type": model_type, "task": task,
            "supported_tasks": tasks, "reason": "",
        }
    return {
        "ok": True, "ready": True, "state": "task_mismatch",
        "model_type": model_type, "task": task,
        "supported_tasks": tasks,
        "reason": "architecture supported but task not supported",
    }
