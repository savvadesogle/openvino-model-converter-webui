"""Stage pipeline: validate -> download -> export -> compress -> package -> verify -> genai_test.

Emits `@@EVENT stage | payload` markers on stdout for the web UI to consume.
Run with:  python -m ov_converter.pipeline config.json
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import ov_converter.settings as S
from ov_converter import checks, naming, package, versions
from ov_converter.modes import list_modes


@dataclass
class ConvertConfig:
    model_id: str = ""                       # HF id or local path/name
    model_path: str | None = None            # resolved source dir (may be empty)
    dest: str | None = None                  # download destination dir
    download: bool = True
    revision: str | None = None
    token: str | None = None
    task: str = ""                           # auto-detected if empty
    mode: str = "int4_sym"
    group_size: int | None = None
    all_layers: bool = True
    ratio: float | None = None
    backup: str | None = None
    data_aware: dict = field(default_factory=dict)
    only_text: bool = True
    delete_intermediate: bool = True
    output_dir: str | None = None
    intermediate_dir: str | None = None
    run_genai_test: bool = True
    prompt: str | None = None
    genai_device: str = "CPU"
    tfreq_auto_install: bool = False
    keep_fp16_export: bool = False
    download_only: bool = False
    include_only: bool = False
    files: list[str] | None = None
    hf_home: str | None = None
    hf_hub_cache: str | None = None


class Emitter:
    def __init__(self) -> None:
        self.current: str = "validate"

    def emit(self, event: str, stage: str | None, payload: str = "") -> None:
        self.current = stage or self.current
        line = f"@@{event} {self.current} | {payload}".rstrip()
        print(line, flush=True)

    def log(self, text: str, stage: str | None = None) -> None:
        self.emit("LOG", stage or self.current, text)

    def start(self, stage: str) -> None:
        self.emit("STAGE", stage, "start")

    def done(self, stage: str, detail: str = "ok") -> None:
        self.emit("STAGE", stage, f"done {detail}")

    def fail(self, stage: str, message: str) -> None:
        self.emit("STAGE", stage, f"fail {message}")


def load_config(path: str | Path) -> ConvertConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ConvertConfig(**{k: v for k, v in data.items() if k in ConvertConfig.__dataclass_fields__})


def _detect_moe(cfg: dict) -> bool:
    def _has_expert_key(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        return any("expert" in str(k).lower() for k in d) or \
            any(_has_expert_key(v) for v in d.values() if isinstance(v, dict))
    if _has_expert_key(cfg.get("text_config", {})) or _has_expert_key(cfg):
        return True
    arch = (cfg.get("architectures") or [None])[0]
    return "moe" in str(arch).lower() or "moe" in str(cfg.get("model_type", "")).lower()


def resolve_source(cfg: ConvertConfig) -> Path | None:
    if cfg.model_path and Path(cfg.model_path).exists():
        return Path(cfg.model_path)
    from ov_converter import hf
    mid = cfg.model_id
    hf_id = hf.parse_hf_id(mid)
    if hf_id:
        cand = S.model_dir(hf_id)
        if cand.exists():
            return cand
    p = Path(mid)
    if p.exists():
        return p
    return None


def run(cfg: ConvertConfig, emit: Emitter | None = None) -> dict:
    emit = emit or Emitter()
    S.apply_env()
    S.ensure_dirs()
    result: dict = {
        "config": asdict(cfg),
        "versions": versions.versions(),
        "stages": {},
    }

    base = naming.base_name(cfg.model_id or (cfg.model_path or "model"))
    out_dir = Path(cfg.output_dir) if cfg.output_dir else naming.output_dir(base, cfg.mode)
    fp16_dir = Path(cfg.intermediate_dir) if cfg.intermediate_dir else \
        naming.intermediate_dir(base, out_dir.parent)
    if cfg.mode == "none":
        out_dir = fp16_dir

    # ---------------------------------------------------------------- validate
    emit.start("validate")
    try:
        src = resolve_source(cfg)
        if src is None and not cfg.download:
            raise RuntimeError("Source model not found locally and download disabled.")
        mode_info = next((m for m in list_modes() if m.id == cfg.mode), None)
        if mode_info is None or not mode_info.available:
            raise RuntimeError(f"Mode '{cfg.mode}' is not supported by this NNCF version.")
        if cfg.group_size is None:
            cfg.group_size = mode_info.default_group_size
        from ov_converter.hf import read_config
        is_moe = _detect_moe(read_config(src)) if src is not None else False
        errs = checks.validate_convert(cfg.mode, cfg.group_size, cfg.all_layers,
                                       cfg.ratio, cfg.backup, model_is_moe=is_moe)
        if errs:
            raise RuntimeError("; ".join(errs))
        if src is not None:
            emit.done("validate", f"source={src}")
        else:
            emit.done("validate", "ok")
    except Exception as e:  # noqa: BLE001
        emit.fail("validate", str(e))
        return result

    # ---------------------------------------------------------------- download
    emit.start("download")
    try:
        if src is None and cfg.download:
            from ov_converter import hf
            mid = cfg.model_id
            dest = Path(cfg.dest) if cfg.dest else S.model_dir(mid)
            dest.parent.mkdir(parents=True, exist_ok=True)
            rc = hf.download(mid, dest, revision=cfg.revision, token=cfg.token,
                             include_only=cfg.include_only, files=cfg.files,
                             log=lambda t: emit.log(t, "download"),
                             progress=lambda p: emit.emit("PROGRESS", "download", f"{p:.0f}"))
            if rc != 0:
                raise RuntimeError(f"hf download exited with code {rc}")
            src = dest
        if src is None:
            raise RuntimeError("no source resolved")
    except Exception as e:  # noqa: BLE001
        emit.fail("download", str(e))
        return result
    emit.done("download", f"-> {src}")

    # download-only mode: stop here
    if cfg.download_only:
        result["done"] = True
        result["output_dir"] = str(src)
        emit.done("download", f"complete at {src}")
        emit.emit("META", "done", json.dumps(result, ensure_ascii=False, default=str))
        return result

    # refine task from the actual config
    from ov_converter.hf import read_config, task_from_config
    src_cfg = read_config(src)
    if not cfg.task:
        cfg.task = task_from_config(src_cfg)
    result["source"] = str(src)
    result["base"] = base
    result["task"] = cfg.task
    result["output_dir"] = str(out_dir)
    result["mode_token"] = naming.token_for(cfg.mode)

    # ---------------------------------------------------------------- export (dense fp16)
    installed_swapped = False
    if cfg.mode != "none":
        emit.start("tfreq")
        try:
            from ov_converter import tfreq
            tfreq_res = tfreq.required_transformers(read_config(src))
            result["tfreq"] = tfreq_res
            if not tfreq_res.get("ok"):
                if cfg.tfreq_auto_install and tfreq_res.get("recommended"):
                    emit.log("installing transformers " + tfreq_res["recommended"] + " ...", "tfreq")
                    rr = tfreq.install_version(tfreq_res["recommended"],
                                               log=lambda t: emit.log(t, "tfreq"))
                    if not rr.get("ok"):
                        raise RuntimeError("failed to install transformers "
                                           + tfreq_res["recommended"] + ": "
                                           + (rr.get("error") or rr.get("output") or "pip error"))
                    installed_swapped = True
                    result["tfreq"]["installed"] = rr.get("version")
                    emit.done("tfreq", "switched to transformers " + str(rr.get("version")))
                else:
                    raise RuntimeError(tfreq_res.get("reason") or "transformers version mismatch")
            else:
                emit.done("tfreq", "transformers " + str(tfreq_res.get("installed")) + " ok")
        except Exception as e:  # noqa: BLE001
            emit.fail("tfreq", str(e))
            return result
    try:
        # ---------------------------------------------------------------- export (dense fp16) -- always runs
        emit.start("export")
        try:
            from ov_converter.export import export_dense
            rc = export_dense(src, fp16_dir, cfg.task, log=lambda t: emit.log(t, "export"))
            if rc != 0:
                raise RuntimeError(f"optimum export exited with code {rc}")
            emit.done("export", f"-> {fp16_dir}")
        except Exception as e:  # noqa: BLE001
            emit.fail("export", str(e))
            return result

        if cfg.mode != "none":
            # ---------------------------------------------------------------- compress
            emit.start("compress")
            try:
                from ov_converter.compress import compress_dir
                report = compress_dir(
                    fp16_dir, out_dir, mode=cfg.mode, group_size=cfg.group_size,
                    all_layers=cfg.all_layers, ratio=cfg.ratio, backup=cfg.backup,
                    only_text=cfg.only_text, data_aware=cfg.data_aware,
                    log=lambda t: emit.log(t, "compress"))
                result["compress_report"] = report
                failed = [k for k, v in report.items() if v.startswith("fail")]
                if failed:
                    raise RuntimeError(f"compression failed for: {failed}")
                emit.done("compress", str({k: v for k, v in report.items()}))
            except Exception as e:  # noqa: BLE001
                emit.fail("compress", str(e))
                return result

        # ---------------------------------------------------------------- package
        emit.start("package")
        try:
            copied = package.copy_metadata(fp16_dir, out_dir)
            mode_label = next((m.label for m in list_modes() if m.id == cfg.mode), cfg.mode)
            info = {
                "base_model": src_cfg.get("_name_or_path", cfg.model_id) or cfg.model_id,
                "base_org": (cfg.model_id or src.name).split("/")[0] or src.parent.name,
                "output_name": out_dir.name,
                "task": cfg.task,
                "mode": cfg.mode,
                "mode_label": mode_label,
                "mode_token": naming.token_for(cfg.mode),
                "bits": next((m.bits for m in list_modes() if m.id == cfg.mode), None),
                "group_size": cfg.group_size,
                "ratio": cfg.ratio,
                "backup": cfg.backup or "none",
                "ignored_scope": None,
                "license": src_cfg.get("license") or "other",
                "versions": versions.versions(),
            }
            package.write_manifest(out_dir, info)
            package.generate_readme(out_dir, info)
            emit.done("package", f"copied {len(copied)} files")
        except Exception as e:  # noqa: BLE001
            emit.fail("package", str(e))
            return result

        # ---------------------------------------------------------------- tokenizer check
        emit.start("tokenizer")
        try:
            v = package.verify(out_dir)
            if not v["ok"]:
                raise RuntimeError(f"tokenizer/config missing: {v['missing_tokenizer']}")
            result["verify"] = v
            emit.done("tokenizer", "tokenizer + config present")
        except Exception as e:  # noqa: BLE001
            emit.fail("tokenizer", str(e))
            return result

        # ---------------------------------------------------------------- cleanup intermediate
        if cfg.delete_intermediate and not cfg.keep_fp16_export \
                and fp16_dir != out_dir and fp16_dir.exists():
            shutil.rmtree(fp16_dir, ignore_errors=True)
            emit.log("intermediate fp16 export removed", "package")

        # ---------------------------------------------------------------- genai test
        if cfg.run_genai_test:
            emit.start("genai_test")
            try:
                from ov_converter.genai_test import run_test, format_result
                res = run_test(out_dir, prompt=cfg.prompt, device=cfg.genai_device,
                               log=lambda t: emit.log(t, "genai_test"))
                result["genai_test"] = res
                emit.log(format_result(res), "genai_test")
                emit.done("genai_test", f"{res['tok_per_s']} tok/s")
            except Exception as e:  # noqa: BLE001
                result["genai_test"] = {"ok": False, "error": str(e)}
                emit.fail("genai_test", str(e))
    finally:
        if cfg.tfreq_auto_install and installed_swapped:
            emit.log("restoring pinned transformers version ...", "tfreq")
            try:
                from ov_converter import tfreq
                tfreq.restore(log=lambda t: emit.log(t, "tfreq"))
            except Exception:  # noqa: BLE001
                emit.log("transformers restore failed (non-fatal)", "tfreq")

    result["done"] = True
    emit.emit("META", "done", json.dumps(result, ensure_ascii=False, default=str))
    return result


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m ov_converter.pipeline <config.json>", file=sys.stderr)
        return 2
    cfg = load_config(sys.argv[1])
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
