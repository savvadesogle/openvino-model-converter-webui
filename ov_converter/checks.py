"""Disk, virtual-memory (Windows pagefile) and conversion-parameter checks."""
from __future__ import annotations

import ctypes
import json
import shutil
import struct
from pathlib import Path

import ov_converter.settings as S


# ---------------------------------------------------------------- disk
def disk_free(path: str | Path) -> int:
    return shutil.disk_usage(str(path)).free


def disk_usage(path: str | Path) -> tuple[int, int, int]:
    u = shutil.disk_usage(str(path))
    return u.total, u.used, u.free


def dir_size(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def disk_check(path: str | Path, needed: int) -> dict:
    """(ok, free, needed) for a target path."""
    free = disk_free(path)
    return {"ok": free >= needed, "free_gb": round(free / 1e9, 1),
            "needed_gb": round(needed / 1e9, 1)}


# ---------------------------------------------------------------- ram / pagefile (Windows)
class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def virtual_memory() -> dict | None:
    """Windows virtual memory incl. pagefile (via GlobalMemoryStatusEx)."""
    if not hasattr(ctypes, "windll"):
        return None
    try:
        m = _MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        if not ok:
            return None
        return {
            "total_phys_gb": round(m.ullTotalPhys / 1e9, 1),
            "avail_phys_gb": round(m.ullAvailPhys / 1e9, 1),
            "total_virtual_gb": round(m.ullTotalVirtual / 1e9, 1),
            "avail_virtual_gb": round(m.ullAvailVirtual / 1e9, 1),
            "total_pagefile_gb": round(m.ullTotalPageFile / 1e9, 1),
            "avail_pagefile_gb": round(m.ullAvailPageFile / 1e9, 1),
        }
    except Exception:  # noqa: BLE001
        return None


def ram_check(needed_bytes: int) -> dict:
    """Check that the virtual address space (phys + pagefile) can hold `needed`."""
    vm = virtual_memory()
    if vm is None:
        return {"ok": True, "reason": "non-Windows, skipped",
                "needed_gb": round(needed_bytes / 1e9, 1), "virtual_memory": None}
    avail = vm["avail_virtual_gb"]
    ok = avail * 1e9 >= needed_bytes
    return {"ok": ok, "avail_gb": avail, "needed_gb": round(needed_bytes / 1e9, 1),
            "reason": "avail virtual memory (phys+pagefile) vs peak estimate",
            "virtual_memory": vm}


# ---------------------------------------------------------------- estimates
def params_from_index(model_dir: str | Path) -> int | None:
    """Total params (approx) from model.safetensors.index.json metadata.total_size."""
    idx = Path(model_dir) / "model.safetensors.index.json"
    if idx.exists():
        try:
            meta = json.loads(idx.read_text()).get("metadata", {})
            total = meta.get("total_size")
            if total:
                # bf16 -> 2 bytes per param
                return int(total) // 2
        except Exception:  # noqa: BLE001
            pass
    return None


def estimate_download_bytes(model_info) -> int:
    """Sum of LFS siblings sizes from a HF ModelInfo."""
    total = 0
    for s in getattr(model_info, "siblings", []) or []:
        if getattr(s, "lfs", None):
            total += s.lfs.get("size", 0)
    return total


def estimate_convert_needed(params: int | None, mode_bits: int | None,
                           keep_source: bool = True) -> int:
    """Peak bytes during conversion: fp16 IR + compressed result (+ source)."""
    if params is None:
        return 0
    fp16 = params * 2
    bits = mode_bits if mode_bits else 16
    result = params * (bits / 8)
    total = fp16 * 1.15 + result * 1.1
    if keep_source:
        total += fp16  # source bf16 stays on disk
    return int(total)


def estimate_ram_needed(params: int | None) -> int:
    """Peak RAM for NNCF compress on a dense fp16 IR (~2x the IR size)."""
    if params is None:
        return 0
    return int(params * 2 * 2 * 1.2)


# ---------------------------------------------------------------- param validation
def validate_convert(mode_id: str, group_size: int, all_layers: bool,
                     ratio, backup, model_is_moe: bool) -> list[str]:
    errors: list[str] = []
    m = next((x for x in __import__("ov_converter.modes", fromlist=["x"]).list_modes()
              if x.id == mode_id), None)
    if m is None:
        return [f"Unknown mode: {mode_id}"]
    if m.moe_only and not model_is_moe:
        errors.append("This mode requires a Mixture-of-Experts model.")
    if m.requires_per_channel and group_size != -1:
        errors.append(f"{mode_id} requires per-channel quantization (group_size = -1).")
    if m.group_size_fixed and group_size != m.group_size_fixed:
        errors.append(f"{mode_id} has a fixed group size of {m.group_size_fixed}.")
    if mode_id == "none":
        return errors
    if group_size not in m.group_size_choices:
        errors.append(f"group_size {group_size} not in allowed {m.group_size_choices}.")
    if mode_id in ("int2_mix", "int3_mix") and ratio not in (None, 1.0):
        errors.append("Mixed two-pass mode ignores ratio; use a single-pass mode for ratio.")
    return errors


def dir_has_weights(model_dir: str | Path) -> bool:
    d = Path(model_dir)
    if (d / "model.safetensors.index.json").exists() or list(d.glob("*.safetensors")) or \
       list(d.glob("*.bin")):
        return True
    return False
