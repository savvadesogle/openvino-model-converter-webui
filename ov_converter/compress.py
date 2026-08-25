"""NNCF weight compression for OpenVINO IRs (single-mode and two-pass int2/int3-mix)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import openvino as ov
from nncf import BackupMode, CompressWeightsMode, IgnoredScope, compress_weights

from ov_converter import naming

EXPERT_PATTERN = r".*mlp\.experts\..*"

MODE_ENUM = {
    "int8_sym": CompressWeightsMode.INT8_SYM,
    "int8_asym": CompressWeightsMode.INT8_ASYM,
    "int4_sym": CompressWeightsMode.INT4_SYM,
    "int4_asym": CompressWeightsMode.INT4_ASYM,
    "int3_sym": CompressWeightsMode.INT3_SYM,
    "int2_sym": CompressWeightsMode.INT2_SYM,
    "nf4": CompressWeightsMode.NF4,
    "mxfp4": CompressWeightsMode.MXFP4,
    "mxfp8_e4m3": CompressWeightsMode.MXFP8_E4M3,
    "fp8_e4m3": CompressWeightsMode.FP8_E4M3,
    "cb4": CompressWeightsMode.CB4,
}

BACKUP_MODE = {
    "none": BackupMode.NONE,
    "int8_sym": BackupMode.INT8_SYM,
    "int8_asym": BackupMode.INT8_ASYM,
    "fp8_e4m3": BackupMode.FP8_E4M3,
    "mxfp8_e4m3": BackupMode.MXFP8_E4M3,
}


def _log_lines(log: Callable[[str], None], text: str) -> None:
    if log:
        log(text)


def compress_ir(ir_path: str | Path, out_path: str | Path, *, mode: str,
                group_size: int, all_layers: bool = True, ratio: float | None = None,
                backup: str | None = None, ignore_patterns: list[str] | None = None,
                log: Callable[[str], None] | None = None,
                data_aware: dict | None = None) -> None:
    """Compress a single OpenVINO IR and save it."""
    ir_path = Path(ir_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log_lines(log, f"Reading {ir_path}")
    model = ov.Core().read_model(ir_path)

    if mode in ("int2_mix", "int3_mix"):
        bits = 2 if mode == "int2_mix" else 3
        model = _two_pass_mix(model, bits, group_size, log)
    elif mode == "none":
        pass
    else:
        is_int8 = mode in ("int8_sym", "int8_asym")
        kwargs = dict(
            mode=MODE_ENUM[mode],
            group_size=group_size,
        )
        if is_int8:
            kwargs["all_layers"] = None
            kwargs["backup_mode"] = None
        else:
            kwargs["all_layers"] = all_layers
            if backup == "none":
                kwargs["backup_mode"] = BackupMode.NONE
            elif backup:
                kwargs["backup_mode"] = BACKUP_MODE.get(backup)
        if ratio is not None:
            kwargs["ratio"] = ratio
        if ignore_patterns:
            kwargs["ignored_scope"] = IgnoredScope(patterns=ignore_patterns)
        if data_aware and not is_int8:
            kwargs.update(_data_aware_kwargs(data_aware))
        eff_all_layers = None if is_int8 else all_layers
        eff_backup = None if is_int8 else backup
        _log_lines(log, f"compress_weights(mode={mode}, group_size={group_size}, "
                        f"all_layers={eff_all_layers}, ratio={ratio}, backup={eff_backup})")
        model = compress_weights(model, **kwargs)

    ov.save_model(model, str(out_path), compress_to_fp16=False)
    _log_lines(log, f"Saved {out_path}")


def _two_pass_mix(model, bits: int, expert_group_size: int,
                  log: Callable[[str], None]) -> "ov.Model":
    """Pass 1: int4 on everything except routed experts. Pass 2: int2/int3 on experts."""
    base_mode = CompressWeightsMode.INT4_SYM
    expert_mode = CompressWeightsMode.INT2_SYM if bits == 2 else CompressWeightsMode.INT3_SYM
    base_gs = 128

    _log_lines(log, f"Pass 1: {base_mode.value} g{base_gs} on non-expert layers")
    m1 = compress_weights(model, mode=base_mode, group_size=base_gs,
                          ignored_scope=IgnoredScope(patterns=[EXPERT_PATTERN]),
                          all_layers=True)

    non_expert = sorted(
        n.get_friendly_name()
        for n in m1.get_ops()
        if n.get_type_name() == "MatMul" and not re.search(EXPERT_PATTERN, n.get_friendly_name())
    )
    _log_lines(log, f"Non-expert MatMuls protected: {len(non_expert)}")
    _log_lines(log, f"Pass 2: {expert_mode.value} g{expert_group_size} on routed experts")
    m2 = compress_weights(m1, mode=expert_mode, group_size=expert_group_size,
                          ignored_scope=IgnoredScope(names=non_expert),
                          all_layers=True)
    return m2


def _data_aware_kwargs(da: dict) -> dict:
    kwargs = {}
    dataset_path = da.get("dataset")
    num_samples = da.get("num_samples", 128)
    if dataset_path:
        from nncf import Dataset
        import numpy as np

        arr = np.load(dataset_path) if str(dataset_path).endswith(".npy") else None
        if arr is None:
            raise ValueError("dataset must be a .npy file of calibration inputs")
        n = int(num_samples)
        n = max(1, min(n, len(arr)))
        if arr.ndim == 1:
            items = [np.array([v]) for v in arr[:n]]
        else:
            items = [arr[i] for i in range(n)]
        kwargs["dataset"] = Dataset(items)
        kwargs["subset_size"] = n
    for flag in ("awq", "scale_estimation", "gptq", "lora_correction"):
        if da.get(flag):
            kwargs[flag] = True
    return kwargs


VISION_SUBMODELS = ("openvino_vision_model", "openvino_vision_embeddings_model",
                    "openvino_vision_embeddings_pos_model",
                    "openvino_vision_embeddings_merger_model")


def _copy_pair(src_dir: Path, dst_dir: Path, stem: str, log) -> None:
    for ext in (".xml", ".bin"):
        f = src_dir / f"{stem}{ext}"
        if f.exists():
            (dst_dir / f"{stem}{ext}").write_bytes(f.read_bytes())
            _log_lines(log, f"copied {stem}{ext} (unchanged)")


def compress_dir(src_dir: str | Path, dst_dir: str | Path, *, mode: str,
                 group_size: int, all_layers: bool, ratio: float | None,
                 backup: str | None, only_text: bool,
                 data_aware: dict | None = None,
                 log: Callable[[str], None] | None = None) -> dict:
    """Compress every `openvino_*.xml` submodel from src_dir into dst_dir.

    With `only_text=True`, only language/text submodels are compressed; vision
    submodels and the tokenizer are copied unchanged (kept fp16).
    """
    from ov_converter.export import list_submodels

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    submodels = list_submodels(src_dir)
    _log_lines(log, f"Submodels found: {submodels}")

    if data_aware and mode in ("int8_sym", "int8_asym"):
        _log_lines(log, "data_aware ignored for int8 mode")
        data_aware = None

    # copy non-OpenVINO metadata files (configs, tokenizer sources, etc.),
    # plus the tokenizer/detokenizer IRs (they are not compressed)
    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix in (".xml", ".bin") and f.stem.startswith("openvino_"):
            if f.stem in ("openvino_tokenizer", "openvino_detokenizer"):
                dst_dir.joinpath(f.name).write_bytes(f.read_bytes())
            continue
        dst_dir.joinpath(f.name).write_bytes(f.read_bytes())

    report: dict[str, str] = {}
    for sm in submodels:
        stem = Path(sm).stem
        if only_text and stem in VISION_SUBMODELS:
            _copy_pair(src_dir, dst_dir, stem, log)
            report[sm] = "copied fp16"
            continue
        src = src_dir / sm
        dst = dst_dir / sm
        try:
            compress_ir(src, dst, mode=mode, group_size=group_size,
                        all_layers=all_layers, ratio=ratio, backup=backup,
                        data_aware=data_aware, log=log)
            report[sm] = "ok"
        except Exception as e:  # noqa: BLE001
            report[sm] = f"fail: {e}"
            _log_lines(log, f"FAIL {sm}: {e}")
    return report


def token_for_mode(mode: str) -> str:
    return naming.token_for(mode)
