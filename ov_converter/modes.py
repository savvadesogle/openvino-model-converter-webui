"""Dynamic list of weight-compression modes supported by the installed NNCF for the OV backend."""
from __future__ import annotations

import functools
from dataclasses import dataclass, asdict

import nncf
from nncf import CompressWeightsMode

from ov_converter.naming import MODE_TOKENS, MODE_HELP


@dataclass
class ModeInfo:
    id: str
    label: str
    bits: int | None
    symmetric: bool | None
    default_group_size: int
    group_size_choices: list[int]
    requires_per_channel: bool   # group_size must be -1
    default_ratio: float | None
    backup_precision: str | None
    group_size_fixed: int | None  # fixed (mx formats)
    moe_only: bool
    help: str
    available: bool               # present in the installed NNCF enum


# curated OV-backend support map; availability is checked against the enum at runtime
OV_SUPPORTED: dict[str, dict] = {
    "int8_sym":    dict(bits=8,  sym=True,  dg=-1, choices=[-1], per_channel=True,  fixed=None, moe=False, ratio=None, backup=None),
    "int8_asym":   dict(bits=8,  sym=False, dg=-1, choices=[-1], per_channel=True,  fixed=None, moe=False, ratio=None, backup=None),
    "int4_sym":    dict(bits=4,  sym=True,  dg=128, choices=[-1, 32, 64, 128, 256], per_channel=False, fixed=None, moe=False, ratio=None, backup=None),
    "int4_asym":   dict(bits=4,  sym=False, dg=128, choices=[-1, 32, 64, 128, 256], per_channel=False, fixed=None, moe=False, ratio=None, backup=None),
    "int3_sym":    dict(bits=3,  sym=True,  dg=64, choices=[-1, 32, 64, 128, 256], per_channel=False, fixed=None, moe=False, ratio=None, backup=None),
    "int2_sym":    dict(bits=2,  sym=True,  dg=64, choices=[-1, 32, 64, 128, 256], per_channel=False, fixed=None, moe=False, ratio=None, backup=None),
    "nf4":         dict(bits=4,  sym=None,  dg=64, choices=[-1, 32, 64, 128, 256], per_channel=False, fixed=None, moe=False, ratio=None, backup=None),
    "mxfp4":       dict(bits=4,  sym=None,  dg=32, choices=[32], per_channel=False, fixed=32, moe=False, ratio=None, backup=None),
    "mxfp8_e4m3":  dict(bits=8,  sym=None,  dg=32, choices=[32], per_channel=False, fixed=32, moe=False, ratio=None, backup=None),
    "fp8_e4m3":    dict(bits=8,  sym=None,  dg=64, choices=[-1, 32, 64, 128], per_channel=False, fixed=None, moe=False, ratio=None, backup=None),
    "cb4":         dict(bits=4,  sym=None,  dg=-1, choices=[-1], per_channel=True,  fixed=None, moe=False, ratio=None, backup=None),
    "int2_mix":    dict(bits=2,  sym=True,  dg=64, choices=[-1, 32, 64, 128], per_channel=False, fixed=None, moe=True, ratio=None, backup="int4"),
    "int3_mix":    dict(bits=3,  sym=True,  dg=64, choices=[-1, 32, 64, 128], per_channel=False, fixed=None, moe=True, ratio=None, backup="int4"),
    "none":        dict(bits=None, sym=None, dg=-1, choices=[-1], per_channel=False, fixed=None, moe=False, ratio=None, backup=None),
}

_ENUM_MEMBERS = {m.value: m for m in CompressWeightsMode}


def _enum_available(mode_id: str) -> bool:
    if mode_id in ("int2_mix", "int3_mix", "none"):
        return True  # composite modes, always available
    return mode_id in _ENUM_MEMBERS


def list_modes() -> list[ModeInfo]:
    """All modes the installed NNCF + OpenVINO backend actually support."""
    out = []
    for mid, cfg in OV_SUPPORTED.items():
        out.append(ModeInfo(
            id=mid,
            label=mid.replace("_", " ").upper(),
            bits=cfg["bits"],
            symmetric=cfg["sym"],
            default_group_size=cfg["dg"],
            group_size_choices=cfg["choices"],
            requires_per_channel=cfg["per_channel"],
            default_ratio=None,
            backup_precision=None,
            group_size_fixed=cfg["fixed"],
            moe_only=cfg["moe"],
            help=MODE_HELP[mid],
            available=_enum_available(mid),
        ))
    return out


def modes_dict() -> list[dict]:
    return [asdict(m) for m in list_modes()]


@functools.lru_cache(maxsize=1)
def self_test_all() -> dict[str, str]:
    """Run a tiny compress+compile for every mode; return mode id -> ok/fail."""
    import numpy as np
    import openvino as ov

    result: dict[str, str] = {}
    try:
        w = np.random.normal(0, 0.1, (256, 128)).astype(np.float32)
        inp = ov.opset13.parameter(ov.PartialShape([1, 128]), ov.Type.f32)
        mm = ov.opset13.matmul(inp, ov.opset13.constant(w), False, True)
        model = ov.Model([ov.opset13.result(mm.output(0))], [inp])
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}

    from nncf import compress_weights

    for m in list_modes():
        if not m.available or m.id == "none":
            continue
        if m.id in ("int2_mix", "int3_mix"):
            result[m.id] = "ok(composite)"
            continue
        try:
            gs = None if (m.group_size_fixed is not None) else m.default_group_size
            cm = compress_weights(model.clone(), mode=_ENUM_MEMBERS[m.id],
                                  group_size=gs, all_layers=True)
            ov.Core().compile_model(cm, "CPU")
            result[m.id] = "ok"
        except Exception as e:  # noqa: BLE001
            result[m.id] = f"fail: {str(e)[:80]}"
    return result
