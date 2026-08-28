"""Report which NNCF CompressWeightsMode members exist in the resolved Python env (subprocess), with a disk cache."""
from __future__ import annotations

import importlib.metadata as im
import json
import os
import sys

import ov_converter.settings as S

CACHE_FILE = S.PROJECT_DIR / "logs" / "ov_modes.json"
SIGNATURE_LIBS = ["nncf"]


def _version(dist: str) -> str:
    try:
        return im.version(dist)
    except im.PackageNotFoundError:
        return "?"


def _signature() -> dict:
    sig = {name: _version(name) for name in SIGNATURE_LIBS}
    sig["python"] = sys.version.split()[0]
    return sig


def _load_cache(sig: dict) -> dict | None:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if data.get("signature") != sig:
        return None
    return data


def _save_cache(sig: dict, data: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"signature": sig, "nncf": data["nncf"], "members": data["members"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, CACHE_FILE)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    sig = _signature()
    cached = _load_cache(sig)
    if cached is not None:
        print(json.dumps({"signature": sig, "nncf": cached["nncf"], "members": cached["members"]}))
        return
    try:
        import nncf
        members = sorted(m.value for m in nncf.CompressWeightsMode)
        nncf_version = _version("nncf")
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": repr(e)}))
        return
    _save_cache(sig, {"nncf": nncf_version, "members": members})
    print(json.dumps({"signature": sig, "nncf": nncf_version, "members": members}))


if __name__ == "__main__":
    main()
