"""Quality evaluation: dense fp16-ov baseline vs compressed candidate.

Runs end-to-end smoke (e2e), causal-LM perplexity on a bundled corpus, and a
side-by-side token-overlap check (First/Sum Divergent Tokens) between the two
OpenVINO IR directories.

Emits `@@EVENT stage | payload` markers on stdout for the web UI to consume.
Run with:  python -m ov_converter.eval config.json
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import ov_converter.settings as S

PROJECT_DIR = S.PROJECT_DIR
DEFAULT_CORPUS = PROJECT_DIR / "ov_converter" / "eval_corpus.txt"
VALID_TESTS = ("e2e", "perplexity", "side_by_side")

_FIXED_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Explain the water cycle in a few short sentences.",
    "What is the capital of France and why is it famous?",
    "Write a short story about a robot learning to paint.",
    "Summarize the history of the internet in two sentences.",
    "How does photosynthesis work in simple terms?",
    "Give me three healthy ideas for a simple breakfast.",
    "Describe the four seasons in a temperate climate.",
    "Why does the sky appear blue during the day?",
    "What are the main differences between cats and dogs?",
]

_PROMPT_LIMIT = 200  # chars kept per generated output


class Emitter:
    def __init__(self) -> None:
        self.current: str = "testing"

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


@dataclass
class TestConfig:
    baseline: str = ""                       # dense fp16-ov dir
    candidate: str = ""                      # compressed dir
    tests: list[str] = field(default_factory=lambda: ["e2e"])
    prompt: str | None = None
    max_new_tokens: int = 64
    num_prompts: int = 8
    corpus: str | None = None                # optional .txt; default bundled eval_corpus.txt
    device: str = "CPU"

    def __post_init__(self) -> None:
        self.tests = [t for t in self.tests if t in VALID_TESTS] or ["e2e"]


def load_config(path: str | Path) -> TestConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TestConfig(**{k: v for k, v in data.items() if k in TestConfig.__dataclass_fields__})


_cfg: TestConfig = TestConfig()


def _clean(s: str) -> str:
    return s.encode("utf-8", errors="replace").decode("utf-8")


def corpus_lines() -> list[str]:
    path = Path(_cfg.corpus) if _cfg.corpus else DEFAULT_CORPUS
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def e2e(model_dir: str | Path, *, log=None) -> dict:
    """End-to-end GenAI smoke on a text OR VLM dir (reuses genai_test.run_test)."""
    from ov_converter.genai_test import run_test
    return run_test(model_dir, prompt=_cfg.prompt, max_new_tokens=_cfg.max_new_tokens,
                    device=_cfg.device, log=log)


def perplexity(model_dir: str | Path, *, log=None) -> dict:
    """Causal-LM perplexity of a text-only OV dir over the eval corpus."""
    d = Path(model_dir)
    if (d / "openvino_vision_embeddings_model.xml").exists() or \
            not (d / "openvino_model.xml").exists():
        return {"ok": False, "reason": "perplexity requires a text-only model (openvino_model.xml)"}
    try:
        import torch
        import torch.nn.functional as F
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        if log:
            log(f"loading OVModelForCausalLM from {d.name}")
        model = OVModelForCausalLM.from_pretrained(str(d), compile=True, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(str(d))
        if log:
            log(f"model compiled, tokenizer ready ({d.name})")
        lines = [ln for ln in corpus_lines() if len(ln) >= 20]
        total_nll = 0.0
        total_tokens = 0
        done = 0
        with torch.no_grad():
            for ln in lines:
                enc = tok(ln, return_tensors="pt", truncation=True, max_length=512)
                ids = enc["input_ids"]
                if ids.shape[1] < 2:
                    continue
                out = model(**enc)
                shift = out.logits[:, :-1, :].reshape(-1, out.logits.shape[-1])
                labels = ids[:, 1:].reshape(-1)
                valid = labels < out.logits.shape[-1]
                labels = labels.clamp(max=out.logits.shape[-1] - 1)
                nll = F.cross_entropy(shift, labels, reduction="none") * valid
                total_nll += float(nll.sum())
                total_tokens += int(valid.sum())
                done += 1
                if log and done % 50 == 0:
                    log(f"perplexity progress: {done}/{len(lines)} lines")
        del model
        gc.collect()
        if total_tokens == 0:
            return {"ok": False, "reason": "no tokens processed from corpus"}
        return {
            "ok": True,
            "ppl": round(math.exp(total_nll / total_tokens), 2),
            "nll": round(total_nll, 3),
            "tokens": total_tokens,
            "lines": done,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:300]}


def side_by_side(baseline: str | Path, candidate: str | Path, *, log=None) -> dict:
    """Token-level First/Sum Divergent Tokens between baseline and candidate.

    Text-only dirs use openvino_genai.LLMPipeline (greedy). VLM dirs fall back
    to genai_test.run_test outputs. Models are loaded sequentially to limit RAM.
    """
    from transformers import AutoTokenizer

    d_b, d_c = Path(baseline), Path(candidate)
    prompts = _FIXED_PROMPTS[:_cfg.num_prompts] or _FIXED_PROMPTS
    is_vlm = (d_b / "openvino_vision_embeddings_model.xml").exists() or \
        (d_c / "openvino_vision_embeddings_model.xml").exists()

    if is_vlm:
        from ov_converter.genai_test import run_test
        if log:
            log("VLM dirs detected -> falling back to genai_test outputs")
        base_outs = [run_test(baseline, prompt=p, max_new_tokens=_cfg.max_new_tokens,
                              device=_cfg.device)["output"] for p in prompts]
        cand_outs = [run_test(candidate, prompt=p, max_new_tokens=_cfg.max_new_tokens,
                              device=_cfg.device)["output"] for p in prompts]
    else:
        import openvino_genai as genai
        if log:
            log(f"compiling baseline LLMPipeline ({_cfg.device}) ...")
        pipe = genai.LLMPipeline(str(d_b), _cfg.device)
        base_outs = [_clean(str(pipe.generate(p, max_new_tokens=_cfg.max_new_tokens, do_sample=False)))
                     for p in prompts]
        del pipe
        gc.collect()
        if log:
            log(f"compiling candidate LLMPipeline ({_cfg.device}) ...")
        pipe = genai.LLMPipeline(str(d_c), _cfg.device)
        cand_outs = [_clean(str(pipe.generate(p, max_new_tokens=_cfg.max_new_tokens, do_sample=False)))
                     for p in prompts]
        del pipe
        gc.collect()

    tok = AutoTokenizer.from_pretrained(str(d_b))
    rows: list[dict] = []
    fdt_vals, sdt_vals, exact_vals = [], [], []
    for p, bo, co in zip(prompts, base_outs, cand_outs):
        bt = tok(bo, add_special_tokens=False)["input_ids"]
        ct = tok(co, add_special_tokens=False)["input_ids"]
        L = max(len(bt), 1)
        n = min(len(bt), len(ct))
        fdt = next((i for i in range(n) if bt[i] != ct[i]), n)
        sdt = sum(1 for i in range(n) if bt[i] != ct[i]) + abs(len(bt) - len(ct))
        exact = 1 if bt == ct else 0
        fdt_n = round(fdt / L, 3)
        sdt_n = round(sdt / L, 3)
        fdt_vals.append(fdt_n)
        sdt_vals.append(sdt_n)
        exact_vals.append(exact)
        rows.append({
            "prompt": p,
            "baseline": _clean(bo)[:_PROMPT_LIMIT],
            "candidate": _clean(co)[:_PROMPT_LIMIT],
            "fdt": fdt_n,
            "sdt": sdt_n,
            "exact": exact,
        })
    nprompts = max(len(fdt_vals), 1)
    return {
        "prompts": rows,
        "avg_fdt": round(sum(fdt_vals) / nprompts, 3),
        "avg_sdt": round(sum(sdt_vals) / nprompts, 3),
        "exact_match_pct": round(sum(exact_vals) / nprompts * 100, 2),
    }


def _done_detail(name: str, res: dict) -> str:
    if name == "e2e":
        return f"{res.get('tok_per_s')} tok/s"
    if name == "perplexity":
        sub = res.get("candidate") if isinstance(res.get("candidate"), dict) else res
        return f"ppl={sub.get('ppl')} over {sub.get('lines')} lines"
    if name == "side_by_side":
        return f"avg_fdt={res.get('avg_fdt')} exact={res.get('exact_match_pct')}%"
    return "ok"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m ov_converter.eval <config.json>", file=sys.stderr)
        return 2
    S.apply_env()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    global _cfg
    _cfg = load_config(sys.argv[1])
    emit = Emitter()
    emit.start("testing")
    result: dict = {"baseline": _cfg.baseline, "candidate": _cfg.candidate, "tests": {}}
    rc = 0
    log = lambda t: emit.log(t, "testing")

    for key, d in (("baseline", _cfg.baseline), ("candidate", _cfg.candidate)):
        if d and not Path(d).exists():
            result[key + "_error"] = f"model dir not found: {d}"
            emit.fail("testing", f"{key} model dir not found: {d}")
            rc = 1

    emit.start("corpus")
    try:
        lines = corpus_lines()
        result["corpus"] = {
            "file": str(Path(_cfg.corpus) if _cfg.corpus else DEFAULT_CORPUS),
            "lines": len(lines),
        }
        emit.done("corpus", f"{len(lines)} lines")
    except Exception as e:  # noqa: BLE001
        result["corpus"] = {"ok": False, "error": str(e)[:300]}
        emit.fail("corpus", str(e)[:300])
        rc = 1

    for name in _cfg.tests:
        emit.start(name)
        try:
            if name == "e2e":
                res = e2e(_cfg.candidate, log=log)
            elif name == "perplexity":
                res = {
                    "baseline": perplexity(_cfg.baseline, log=log),
                    "candidate": perplexity(_cfg.candidate, log=log),
                }
                res["ok"] = res["baseline"].get("ok") and res["candidate"].get("ok")
            elif name == "side_by_side":
                res = side_by_side(_cfg.baseline, _cfg.candidate, log=log)
                res["ok"] = True
            else:
                raise ValueError(f"unknown test: {name}")
            result["tests"][name] = res
            if res.get("ok"):
                emit.done(name, _done_detail(name, res))
            else:
                msg = res.get("reason") or res.get("error") or "failed"
                for side in ("baseline", "candidate"):
                    sub = res.get(side)
                    if isinstance(sub, dict) and not sub.get("ok"):
                        msg = sub.get("reason") or sub.get("error") or msg
                emit.fail(name, str(msg)[:200])
        except Exception as e:  # noqa: BLE001
            result["tests"][name] = {"ok": False, "error": str(e)[:300]}
            emit.fail(name, str(e)[:200])

    emit.done("testing", f"rc={rc}")
    emit.emit("META", "done", json.dumps(result, ensure_ascii=False, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())