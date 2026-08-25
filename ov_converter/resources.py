"""Pre-download resource feasibility: disk and RAM estimates per pipeline stage."""
from __future__ import annotations

from pathlib import Path

import ov_converter.checks as checks
import ov_converter.settings as S

SEVERITY = {"fail": 3, "warn": 2, "unknown": 1, "ok": 0}


def _disk_free_existing(path: str) -> int | None:
    p = Path(path).expanduser()
    if not p.exists():
        while p != p.parent and not p.exists():
            p = p.parent
    if not p.exists():
        return None
    try:
        return checks.disk_free(str(p))
    except Exception:  # noqa: BLE001
        return None


def _gb(b: int) -> float:
    return round(b / 1e9, 1)


def _stage(state: str, ok: bool, need_disk_gb, free_disk_gb, need_ram_gb,
           avail_ram_gb, result_gb, estimated: bool, issue: str | None) -> dict:
    return {
        "state": state,
        "ok": ok,
        "need_disk_gb": need_disk_gb,
        "free_disk_gb": free_disk_gb,
        "need_ram_gb": need_ram_gb,
        "avail_ram_gb": avail_ram_gb,
        "result_gb": result_gb,
        "estimated": estimated,
        "issue": issue,
    }


def _download_stage(orig: int, dl_need: int, free: int | None, path: str,
                    estimated: bool) -> dict:
    try:
        need_gb = _gb(dl_need) if dl_need else None
        free_gb = _gb(free) if free is not None else None
        if orig == 0:
            return _stage("unknown", False, need_gb, free_gb, None, None, None,
                          estimated, "model size unknown")
        if free is None:
            return _stage("unknown", False, need_gb, free_gb, None, None, None,
                          estimated, "disk availability unknown")
        if free < dl_need:
            return _stage("fail", False, need_gb, free_gb, None, None, None,
                          estimated,
                          f"need {need_gb} GB disk, only {free_gb} GB free (on {path})")
        if free < dl_need * 1.25:
            return _stage("warn", True, need_gb, free_gb, None, None, None,
                          estimated,
                          f"disk tight (need {need_gb} GB, free {free_gb} GB)")
        return _stage("ok", True, need_gb, free_gb, None, None, None, estimated, None)
    except Exception:
        return _stage("unknown", False, None, None, None, None, None, estimated,
                      "unexpected error")


def _export_stage(params: int | None, disk_need: int, ram_need: int,
                  free: int | None, avail_ram: int | None, estimated: bool,
                  result_gb: float | None = None) -> dict:
    try:
        if params is None:
            return _stage("unknown", False, None, None, None, None, result_gb,
                          False, "param count unknown — cannot estimate")
        need_disk_gb = _gb(disk_need)
        need_ram_gb = _gb(ram_need)
        free_gb = _gb(free) if free is not None else None
        avail_gb = _gb(avail_ram) if avail_ram is not None else None
        if free is None or avail_ram is None:
            return _stage("unknown", False, need_disk_gb, free_gb, need_ram_gb,
                          avail_gb, result_gb, estimated,
                          "disk or ram availability unknown")
        fails = []
        if free < disk_need:
            fails.append(f"disk: need {need_disk_gb} GB / free {free_gb} GB")
        if avail_ram < ram_need:
            fails.append(f"ram: need {need_ram_gb} GB / avail {avail_gb} GB")
        if fails:
            return _stage("fail", False, need_disk_gb, free_gb, need_ram_gb,
                          avail_gb, result_gb, estimated, "; ".join(fails))
        warnings = []
        if free < disk_need * 1.25 or avail_ram < ram_need * 1.25:
            warnings.append("tight")
        if estimated:
            warnings.append("param count estimated")
        if warnings:
            return _stage("warn", True, need_disk_gb, free_gb, need_ram_gb,
                          avail_gb, result_gb, estimated, "; ".join(warnings))
        return _stage("ok", True, need_disk_gb, free_gb, need_ram_gb, avail_gb,
                      result_gb, estimated, None)
    except Exception:
        return _stage("unknown", False, None, None, None, None, result_gb,
                      estimated, "unexpected error")


def _recommendations(stages: dict, params, output_path: str,
                     download_path: str) -> list[str]:
    recs: list[str] = []
    for name, st in stages.items():
        if st["state"] != "fail":
            continue
        if name == "download":
            recs.append(f"Not enough disk on {download_path} — free up space or pick another drive.")
        else:
            need_ram = st.get("need_ram_gb")
            avail_ram = st.get("avail_ram_gb")
            if need_ram is not None and avail_ram is not None and avail_ram < need_ram:
                recs.append(
                    f"Not enough memory (need {need_ram} GB, available {avail_ram} GB) "
                    f"— close other apps or increase the Windows pagefile / Linux swap.")
            else:
                need_disk = st.get("need_disk_gb")
                free_disk = st.get("free_disk_gb")
                if need_disk is not None and free_disk is not None and free_disk < need_disk:
                    recs.append(f"Not enough disk on {output_path} — free up space or pick another drive.")
        if len(recs) >= 3:
            break
    if not recs and params is None:
        recs.append("Param count unknown — install psutil or provide "
                    "model.safetensors.index.json for a precise estimate.")
    return recs


def analyze(params: int | None = None,
            size_bytes: int = 0,
            mode_bits: int | None = None,
            est_bits: int | None = None,
            download_path: str | None = None,
            output_path: str | None = None,
            group_size: int | None = None,
            scale_bits: int | None = None) -> dict:
    if params and params > 0:
        params = int(params)
        estimated = False
    elif size_bytes and size_bytes > 0:
        params = max(1, int(size_bytes / 2))
        estimated = True
    else:
        params = None
        estimated = False

    download_path = download_path or str(S.OUTPUT_ROOT)
    output_path = output_path or str(S.OUTPUT_ROOT)

    free_dl: int | None = None
    try:
        free_dl = _disk_free_existing(download_path)
    except Exception:
        free_dl = None

    free_out: int | None = None
    try:
        free_out = _disk_free_existing(output_path)
    except Exception:
        free_out = None

    avail_ram: int | None = None
    try:
        vm = checks.virtual_memory()
        if vm is not None:
            avail_ram = int(vm["avail_virtual_gb"] * 1e9)
    except Exception:
        avail_ram = None

    orig = size_bytes
    res_bits = mode_bits if mode_bits else 4
    calc_bits = est_bits if est_bits is not None and est_bits != res_bits else res_bits
    gs = group_size if group_size is not None else 128
    sb = scale_bits if scale_bits is not None else 16
    scale_overhead = (sb / gs) if (gs and gs > 0) else 0.0
    fp16 = params * 2 if params else 0
    res_bytes = int(params * (calc_bits + scale_overhead) / 8 * 1.15) if params else 0
    dl_need = int(orig * 1.05) if orig else 0
    conv_disk_need = int(fp16 * 1.05 + fp16 * 1.15)
    comp_disk_need = int(fp16 * 1.15 + res_bytes)
    ram_export = int(params * 4.8) if params else 0
    ram_compress = int(params * 3.0) if params else 0

    result_gb = _gb(res_bytes) if params else None

    stages = {
        "download": _download_stage(orig, dl_need, free_dl, download_path, estimated),
        "convert": _export_stage(params, conv_disk_need, ram_export, free_out,
                                 avail_ram, estimated),
        "compress": _export_stage(params, comp_disk_need, ram_compress, free_out,
                                  avail_ram, estimated, result_gb),
    }

    overall = max((s["state"] for s in stages.values()),
                  key=lambda st: SEVERITY[st])

    return {
        "params": params,
        "estimated_params": estimated,
        "size_gb": round(size_bytes / 1e9, 2),
        "mode_bits": res_bits,
        "group_size": gs,
        "scale_bits": sb,
        "stages": stages,
        "overall": overall,
        "recommendations": _recommendations(stages, params, output_path,
                                            download_path),
    }