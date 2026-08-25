"""Single-task manager: runs the pipeline subprocess, streams logs, allows cancel."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import ov_converter.settings as S

PROJECT_DIR = S.PROJECT_DIR


class Task:
    def __init__(self, kind: str, config: dict, task_id: str | None = None):
        self.id = task_id or uuid.uuid4().hex[:12]
        self.kind = kind
        self.config = config
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.returncode: int | None = None
        self.error: str | None = None
        self.lines: list[str] = []
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    # -- log stream -----------------------------------------------------
    def append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)

    def since(self, idx: int) -> list[str]:
        with self._lock:
            return self.lines[idx:]

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def is_done(self) -> bool:
        return self.finished_at is not None

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        # persist the task config WITHOUT the token; pass the real token via env
        redacted = dict(self.config)
        if redacted.get("token"):
            redacted["token"] = None
        cfg_path = PROJECT_DIR / "logs" / f"task_{self.id}.json"
        cfg_path.write_text(json.dumps(redacted, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env.setdefault(S.HF_HOME_ENV, str(S.CACHE_ROOT))
        env.setdefault(S.HF_HUB_CACHE_ENV, str(S.CACHE_ROOT / "hub"))
        if self.config.get("hf_home"):
            env[S.HF_HOME_ENV] = str(self.config["hf_home"])
        if self.config.get("hf_hub_cache"):
            env[S.HF_HUB_CACHE_ENV] = str(self.config["hf_hub_cache"])
        if self.config.get("token"):
            env["HF_TOKEN"] = self.config["token"]
        cmd = [S.resolve_python(), "-m", "ov_converter.pipeline", str(cfg_path)]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_DIR), env=env)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            self.append(line.rstrip("\n"))
        rc = self._proc.wait()
        self.returncode = rc
        self.finished_at = time.time()
        if rc != 0 and not self.error:
            self.error = f"pipeline exited with code {rc}"

    def cancel(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.5)
            if self._proc.poll() is None:
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self.error = "cancelled by user"


class TaskManager:
    def __init__(self) -> None:
        self.current: Task | None = None
        self.history: list[Task] = []
        self._lock = threading.Lock()

    def is_busy(self) -> bool:
        return self.current is not None and self.current.is_running()

    def start(self, kind: str, config: dict) -> Task:
        with self._lock:
            if self.is_busy():
                raise RuntimeError("Another task is already running.")
            task = Task(kind, config)
            self.current = task
            self.history.append(task)
        task.start()
        return task

    def cancel(self) -> None:
        t = self.current
        if t:
            t.cancel()

    def status(self) -> dict:
        t = self.current
        if t is None:
            return {"busy": False, "task": None}
        return {
            "busy": t.is_running(),
            "done": t.is_done(),
            "task": {
                "id": t.id,
                "kind": t.kind,
                "started_at": t.started_at,
                "finished_at": t.finished_at,
                "returncode": t.returncode,
                "error": t.error,
                "line_count": len(t.since(0)),
                "config": t.config,
            },
        }


manager = TaskManager()
