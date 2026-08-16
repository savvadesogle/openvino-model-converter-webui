"""Output naming: `<Base>-<mode>-ov`, `-ov` is always last (HF OpenVINO style)."""
from __future__ import annotations

from pathlib import Path

import ov_converter.settings as S

# mode id -> token used in the folder name (matches HF OpenVINO convention)
MODE_TOKENS = {
    "int8_sym": "int8",
    "int8_asym": "int8",
    "int4_sym": "int4",
    "int4_asym": "int4",
    "int3_sym": "int3",
    "int2_sym": "int2",
    "nf4": "nf4",
    "mxfp4": "mxfp4",
    "mxfp8_e4m3": "mxfp8",
    "fp8_e4m3": "fp8e4m3",
    "cb4": "cb4",
    "int2_mix": "int2-mix",
    "int3_mix": "int3-mix",
    "none": "fp16",
}

MODE_HELP = {
    "int2_sym": "2-bit symmetric integer, group quantization.",
    "int3_sym": "3-bit symmetric integer, group quantization.",
    "int4_sym": "4-bit symmetric integer, no zero point.",
    "int4_asym": "4-bit asymmetric integer, with zero point.",
    "int8_sym": "8-bit symmetric, per-channel (group_size must be -1).",
    "int8_asym": "8-bit asymmetric, per-channel (group_size must be -1).",
    "nf4": "4-bit NormalFloat (QLoRA-style), group quantization.",
    "mxfp4": "MX-compliant FP4 (E2M1) with E8M0 scale, group size 32.",
    "mxfp8_e4m3": "MX-compliant FP8 (E4M3) with E8M0 scale, group size 32.",
    "fp8_e4m3": "FP8 (E4M3) with group-level FP16 scale.",
    "cb4": "Codebook, 16 fixed FP8 (E4M3) values, per-channel.",
    "int2_mix": "Two-pass: routed experts at int2, everything else at int4 "
                "(reproduces the AutoRound mixed scheme). MoE models only.",
    "int3_mix": "Two-pass: routed experts at int3, everything else at int4.",
    "none": "No compression - export dense fp16 only.",
}


def token_for(mode: str) -> str:
    return MODE_TOKENS.get(mode, mode)


def base_name(source: str | Path) -> str:
    """Base name of a model from a local path or HF id."""
    p = Path(str(source))
    return p.name


def output_name(base: str, mode: str) -> str:
    """<Base>-<mode>-ov ; strips a trailing -ov if the base already has one."""
    base = base.rstrip("/\\")
    if base.lower().endswith("-ov"):
        base = base[:-3].rstrip("-")
    return f"{base}-{token_for(mode)}-ov"


def intermediate_name(base: str) -> str:
    base = base.rstrip("/\\")
    if base.lower().endswith("-ov"):
        base = base[:-3].rstrip("-")
    return f"{base}{S.INTERMEDIATE_SUFFIX}"


def output_dir(base: str, mode: str, root: str | Path = None) -> Path:
    root = Path(root) if root else S.OUTPUT_ROOT
    return root / output_name(base, mode)


def intermediate_dir(base: str, root: str | Path = None) -> Path:
    root = Path(root) if root else S.OUTPUT_ROOT
    return root / intermediate_name(base)
