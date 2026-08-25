"""Post-conversion packaging: copy metadata, write HF model card + manifest, verify."""
from __future__ import annotations

import json
from pathlib import Path

REQUIRED_TOKENIZER = ("openvino_tokenizer.xml", "openvino_tokenizer.bin",
                      "openvino_detokenizer.xml", "openvino_detokenizer.bin")

# non-tokenizer submodel IR stems produced by optimum export (text/VLM/ASR/vision)
SUBMODEL_STEMS = (
    "openvino_model", "openvino_language_model", "openvino_encoder_model",
    "openvino_decoder_model", "openvino_vision_embeddings_model", "openvino_vision_model",
)


def copy_metadata(src_dir: str | Path, dst_dir: str | Path) -> list[str]:
    """Copy everything that is not an OpenVINO IR/bin (configs, tokenizers, jinja...)."""
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for f in src.iterdir():
        if not f.is_file():
            continue
        if f.suffix in (".xml", ".bin") and f.stem.startswith("openvino_"):
            continue
        dst.joinpath(f.name).write_bytes(f.read_bytes())
        copied.append(f.name)
    return copied


def verify(dst_dir: str | Path) -> dict:
    """Check the converted dir is complete (weights + tokenizer + config)."""
    d = Path(dst_dir)
    missing = [f for f in REQUIRED_TOKENIZER if not (d / f).exists()]
    has_lm = any((d / f"{stem}.xml").exists() for stem in SUBMODEL_STEMS)
    has_config = (d / "config.json").exists()
    return {
        "ok": has_lm and has_config and not missing,
        "language_model": has_lm,
        "config": has_config,
        "missing_tokenizer": missing,
    }


def write_manifest(dst_dir: str | Path, info: dict) -> None:
    info = dict(info)
    if info.get("mode_token") == "fp16" or info.get("mode") == "none":
        info["mode_label"] = "FP16"
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    (Path(dst_dir) / "manifest.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def generate_readme(dst_dir: str | Path, info: dict) -> Path:
    """HF-compatible model card in the style of OpenVINO/*-int4-ov."""
    d = Path(dst_dir)
    d.mkdir(parents=True, exist_ok=True)

    mode = info.get("mode", "?")
    token = info.get("mode_token", mode)
    base = info.get("base_model", "?")
    base_org = info.get("base_org") or base.split("/")[0]
    name = info.get("output_name") or (base.split("/")[-1] + "-" + token + "-ov")
    bits = info.get("bits")
    is_dense = token == "fp16" or mode == "none"
    if is_dense:
        quant_desc = "FP16"
    elif bits:
        quant_desc = f"INT{bits}"
    else:
        quant_desc = mode.upper().replace("_", " ").title()
    group_size = info.get("group_size")
    ratio = info.get("ratio")
    backup = info.get("backup", "none")
    ignored = info.get("ignored_scope") or []
    pipeline_tag = info.get("task", "text-generation")
    license = info.get("license", "unknown")
    ov_ver = info.get("versions", {}).get("openvino", "?")
    optimum_ver = info.get("versions", {}).get("optimum", "?")

    lines = [
        "---",
        f"library_name: transformers",
        f"license: {license}",
        f"pipeline_tag: {pipeline_tag}",
        f"base_model: {base}",
        f"base_model_relation: {'original' if is_dense else 'quantized'}",
        "tags:",
        "  - openvino",
    ]
    if not is_dense:
        lines.append("  - nncf")
    lines += [
        "---",
        f"# {name}",
        "",
        f" * Model creator: [{base_org}](https://huggingface.co/{base_org})",
        f" * Original model: [{base}](https://huggingface.co/{base})",
        "",
        "## Description",
        "",
    ]
    if is_dense:
        lines.append(
            f"This is the [{base}](https://huggingface.co/{base}) model converted to the "
            f"[OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) "
            f"format in **{quant_desc}** precision (dense, no weight compression).")
    else:
        lines.append(
            f"This is the [{base}](https://huggingface.co/{base}) model converted to the "
            f"[OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) "
            f"format with weights compressed to **{quant_desc}** by "
            f"[NNCF](https://github.com/openvinotoolkit/nncf).")
        lines += [
            "",
            "## Quantization Parameters",
            "",
            "Weight compression was performed using `nncf.compress_weights` with the following parameters:",
            "",
            " * mode: **" + (info.get("mode_label") or quant_desc) + "**",
            f" * group_size: **{group_size}**" if group_size is not None else " * group_size: per-channel",
            f" * ratio: **{ratio}**" if ratio is not None else "",
            f" * backup_mode: **{backup}**" if backup not in (None, "none") else "",
        ]
        if ignored:
            lines.append(f" * ignored_scope: layers matching `{', '.join(ignored)}`")
        lines += [
            "",
            "For more information on quantization, check the "
            "[OpenVINO model optimization guide]"
            "(https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html).",
        ]
    lines += [
        "",
        "## Compatibility",
        "",
        f"* OpenVINO version {ov_ver} and higher",
        f"* Optimum Intel {optimum_ver} and higher",
        "",
        "## Conversion manifest",
        "",
        "See `manifest.json` in this directory for the exact commands, versions and settings used.",
        "",
    ]

    (d / "README.md").write_text("\n".join(x for x in lines if x is not None), encoding="utf-8")
    return d / "README.md"
