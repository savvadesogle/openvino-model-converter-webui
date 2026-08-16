"""FastAPI app: two-tab UI (Download / Convert) + SSE task log stream."""
from __future__ import annotations

import asyncio
import json
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
    local = parse_hf_id(body.text) or None
    from ov_converter.hf import is_local_path
    lp = is_local_path(body.text)
    if lp is not None:
        return {"kind": "local", "info": __import__("ov_converter.hf", fromlist=["x"]).detect_local(lp)}
    if not local:
        raise HTTPException(400, "Not a valid HF model id or local path.")
    return {"kind": "hf", "info": validate_model_id(local, body.token)}


@app.post("/api/download")
def api_download(body: DownloadIn):
    if manager.is_busy():
        raise HTTPException(409, "Another task is running.")
    if "/" not in body.model_id:
        raise HTTPException(400, "Expected an HF model id like org/model.")
    dest = S.model_dir(body.model_id)
    task = manager.start("download", {
        "model_id": body.model_id,
        "model_path": None,
        "download": True,
        "download_only": True,
        "revision": body.revision,
        "token": body.token,
        "mode": "none",
        "run_genai_test": False,
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
