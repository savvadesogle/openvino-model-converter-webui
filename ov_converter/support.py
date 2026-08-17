"""Model architecture support lookup for the OpenVINO exporter, computed from installed optimum with a disk cache."""
from __future__ import annotations

import importlib.metadata as im
import json
import sys
import threading
import time

import ov_converter.settings as S

_CACHE_FILE = S.PROJECT_DIR / "logs" / "ov_support.json"
_SIGNATURE_LIBS = ["optimum", "optimum-intel", "openvino", "transformers", "nncf"]

_SUPPORTED: dict | None = None
_ready = threading.Event()
_started = False
_lock = threading.Lock()


def _version(dist: str) -> str:
    try:
        return im.version(dist)
    except im.PackageNotFoundError:
        return "?"


def _signature() -> dict:
    sig = {name: _version(name) for name in _SIGNATURE_LIBS}
    sig["python"] = sys.version.split()[0]
    return sig


def _build_registry() -> dict:
    import optimum.exporters.openvino  # noqa: F401
    from optimum.exporters.tasks import TasksManager

    raw = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES.get("transformers", {})
    unsupported = set(getattr(TasksManager, "_UNSUPPORTED_CLI_MODEL_TYPE", set() or []))
    supported_set = set(TasksManager._SUPPORTED_MODEL_TYPE) - unsupported
    registry = {}
    for mt, fallback in raw.items():
        if mt not in supported_set:
            continue
        try:
            tasks = TasksManager.get_supported_tasks_for_model_type(mt, exporter="openvino", library_name="transformers")
            task_list = sorted(tasks.keys())
        except Exception:  # noqa: BLE001
            if isinstance(fallback, dict):
                task_list = sorted(fallback.keys())
            else:
                task_list = []
        registry[mt] = {"tasks": task_list}
    return dict(sorted(registry.items()))


def _load_cache(sig: dict) -> dict | None:
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if data.get("signature") != sig:
        return None
    return data.get("model_types")


def _save_cache(sig: dict, registry: dict) -> None:
    try:
        data = {"signature": sig, "model_types": registry}
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        pass


def _load() -> dict | None:
    sig = _signature()
    cached = _load_cache(sig)
    if cached is not None:
        return cached
    try:
        registry = _build_registry()
    except Exception:  # noqa: BLE001
        return None
    _save_cache(sig, registry)
    return registry


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
            "reason": "OpenVINO exporter registry unavailable in this environment",
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
