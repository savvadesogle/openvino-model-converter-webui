"""Build the OpenVINO exporter architecture registry inside the resolved Python env (subprocess)."""
from __future__ import annotations

import importlib.metadata as im
import json
import os
import sys

import ov_converter.settings as S

CACHE_FILE = S.PROJECT_DIR / "logs" / "ov_support.json"
SIGNATURE_LIBS = ["optimum", "optimum-intel", "openvino", "transformers", "nncf"]


def _version(dist: str) -> str:
    try:
        return im.version(dist)
    except im.PackageNotFoundError:
        return "?"


def _signature() -> dict:
    sig = {name: _version(name) for name in SIGNATURE_LIBS}
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
            task_list = sorted(fallback.keys()) if isinstance(fallback, dict) else []
        registry[mt] = {"tasks": task_list}
    return dict(sorted(registry.items()))


def _load_cache(sig: dict) -> dict | None:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if data.get("signature") != sig:
        return None
    return data.get("model_types")


def _save_cache(sig: dict, registry: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"signature": sig, "model_types": registry}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, CACHE_FILE)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    sig = _signature()
    cached = _load_cache(sig)
    if cached is not None:
        print(json.dumps({"signature": sig, "model_types": cached}))
        return
    try:
        registry = _build_registry()
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": repr(e)}))
        return
    _save_cache(sig, registry)
    print(json.dumps({"signature": sig, "model_types": registry}))


if __name__ == "__main__":
    main()
