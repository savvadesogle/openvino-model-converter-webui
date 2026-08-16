"""FastAPI app: two-tab UI (Download / Convert) + SSE task log stream."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ov_converter.settings as S
from ov_converter import checks, modes, naming, scan, versions
from ov_converter.hf import parse_hf_id, validate_model_id

from webui.tasks import manager

S.apply_env()
S.ensure_dirs()

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="OpenVINO Model Converter")


# ------------------------------------------------------------------ info
class HfValidateIn(BaseModel):
    text: str
    token: str | None = None


class DownloadIn(BaseModel):
    model_id: str
    revision: str | None = None
    token: str | None = None
    include_only: bool = False
    dest: str | None = None
    files: list[str] | None = None
    hf_home: str | None = None
    hf_hub_cache: str | None = None


class ConvertIn(BaseModel):
    model_id: str
    model_path: str | None = None
    download: bool = True
    revision: str | None = None
    token: str | None = None
    task: str = ""
    mode: str = "int4_sym"
    group_size: int = 128
    all_layers: bool = True
    ratio: float | None = None
    backup: str | None = None
    data_aware: dict = {}
    only_text: bool = True
    delete_intermediate: bool = True
    output_dir: str | None = None
    run_genai_test: bool = True
    prompt: str | None = None
    download_only: bool = False
    hf_home: str | None = None
    hf_hub_cache: str | None = None


class OpenDirIn(BaseModel):
    path: str


class VerifyHashIn(BaseModel):
    path: str
    files: list[dict] = []


class MkdirIn(BaseModel):
    path: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    free = checks.disk_free(S.OUTPUT_ROOT)
    return {
        "paths": {
            "cache": str(S.CACHE_ROOT),
            "originals": str(S.ORIGINALS_ROOT),
            "output": str(S.OUTPUT_ROOT),
            "project": str(S.PROJECT_DIR),
        },
        "disk_free_gb": round(free / 1e9, 1),
        "virtual_memory": checks.virtual_memory(),
        "versions": versions.versions(),
    }


@app.get("/api/modes")
def api_modes():
    return {"modes": modes.modes_dict()}


class MkdirIn(BaseModel):
    path: str


@app.post("/api/mkdir")
def api_mkdir(body: MkdirIn):
    r"""Create a directory, but ONLY under the configured models root (T:\models).

    Guards against accidental writes to other drives / network shares.
    """
    from pathlib import Path

    p = Path(body.path)
    try:
        resolved = p.expanduser().resolve(strict=False)
        root = S.MODELS_ROOT.expanduser().resolve(strict=False)
        if not resolved.is_relative_to(root):
            return {"ok": False,
                    "error": f"refusing to create a directory outside {S.MODELS_ROOT}"}
        p.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": str(p)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.post("/api/modes/self-test")
async def api_self_test():
    result = await asyncio.to_thread(modes.self_test_all)
    return {"result": result}


@app.get("/api/models")
def api_models():
    return {"sources": scan.scan_models(), "converted": scan.scan_converted()}


class EstimateIn(BaseModel):
    path: str


@app.post("/api/model/estimate")
def api_estimate(body: EstimateIn):
    from ov_converter import checks
    return {
        "path": body.path,
        "size_gb": round(checks.dir_size(body.path) / 1e9, 2),
        "params": checks.params_from_index(body.path),
    }


@app.post("/api/hf/validate")
def api_hf_validate(body: HfValidateIn):
    import os
    from ov_converter.hf import is_local_path
    lp = is_local_path(body.text)
    if lp is not None:
        return {"kind": "local", "info": __import__("ov_converter.hf", fromlist=["x"]).detect_local(lp)}
    local = parse_hf_id(body.text)
    if not local:
        raise HTTPException(400, "Not a valid HF model id or local path.")
    token = (body.token or "").strip() or os.environ.get("HF_TOKEN") or None
    return {"kind": "hf", "info": validate_model_id(local, token)}


class LocalCheckIn(BaseModel):
    path: str
    files: list[str] = []


@app.post("/api/model/local-check")
def api_local_check(body: LocalCheckIn):
    from ov_converter.hf import local_check
    return local_check(body.path, body.files)


@app.post("/api/open-dir")
def api_open_dir(body: OpenDirIn):
    path = (body.path or "").strip()
    if not path or not os.path.isdir(path):
        return {"ok": False, "error": "not a directory"}
    try:
        os.startfile(path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    return {"ok": True}


@app.post("/api/mkdir")
def api_mkdir(body: MkdirIn):
    from pathlib import Path
    p = Path(body.path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": str(p)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/model/verify-hash")
def api_verify_hash(body: VerifyHashIn):
    from ov_converter.hf import verify_hashes_stream

    def event_gen():
        for ev in verify_hashes_stream(body.path, body.files):
            yield json.dumps(ev) + "\n"

    return StreamingResponse(event_gen(), media_type="application/x-ndjson")


@app.get("/api/disk")
def api_disk(path: str = ""):
    import shutil
    if not path:
        return {"ok": False, "error": "path is required"}
    try:
        t, u, f = shutil.disk_usage(path)
        return {"ok": True, "path": path, "free_gb": round(f/1e9, 1), "total_gb": round(t/1e9, 1)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/download")
def api_download(body: DownloadIn):
    if manager.is_busy():
        raise HTTPException(409, "Another task is running.")
    if "/" not in body.model_id:
        raise HTTPException(400, "Expected an HF model id like org/model.")
    dest = Path(body.dest) if body.dest else S.model_dir(body.model_id)
    task = manager.start("download", {
        "model_id": body.model_id,
        "model_path": None,
        "dest": str(dest),
        "download": True,
        "download_only": True,
        "revision": body.revision,
        "token": body.token,
        "mode": "none",
        "run_genai_test": False,
        "include_only": body.include_only,
        "files": body.files,
        "hf_home": body.hf_home,
        "hf_hub_cache": body.hf_hub_cache,
    })
    return {"task_id": task.id, "dest": str(dest)}


@app.post("/api/convert")
def api_convert(body: ConvertIn):
    if manager.is_busy():
        raise HTTPException(409, "Another task is running.")
    cfg = body.model_dump()
    task = manager.start("convert", cfg)
    return {"task_id": task.id}


@app.post("/api/task/cancel")
def api_cancel():
    manager.cancel()
    return {"ok": True}


@app.get("/api/task/status")
def api_status():
    return manager.status()


@app.get("/api/task/stream")
async def api_stream(task_id: str | None = None):
    """SSE stream of the current (or a given) task log lines."""
    async def gen():
        t = manager.current
        if t is None:
            yield "event: done\ndata: {}\n\n"
            return
        idx = 0
        while True:
            lines = t.since(idx)
            for ln in lines:
                yield f"data: {json.dumps({'line': ln}, ensure_ascii=False)}\n\n"
            idx += len(lines)
            if not t.is_running():
                # drain the rest then finish
                rest = t.since(idx)
                for ln in rest:
                    yield f"data: {json.dumps({'line': ln}, ensure_ascii=False)}\n\n"
                idx += len(rest)
                yield "event: done\n" + \
                    f"data: {json.dumps({'returncode': t.returncode, 'error': t.error}, ensure_ascii=False)}\n\n"
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
