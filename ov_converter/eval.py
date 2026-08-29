"""Quality evaluation: dense fp16-ov baseline vs compressed candidate.

Runs end-to-end smoke (e2e), causal-LM perplexity on a bundled corpus, and a
side-by-side token-overlap check (First/Sum Divergent Tokens) between the two
OpenVINO IR directories.

Perplexity handles both text-only dirs (openvino_model.xml) and VLM dirs, where
the standalone text decoder (openvino_language_model.xml) plus the in-dir text
embeddings model are loaded and evaluated.

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

    def progress(self, pct: int) -> None:
        pct = max(0, min(100, int(pct)))
        self.emit("PROGRESS", None, str(pct))

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


def _resolve_language_ir(d: Path) -> str | None:
    is_vlm = (d / "openvino_vision_embeddings_model.xml").exists()
    if is_vlm and (d / "openvino_language_model.xml").exists():
        return "openvino_language_model.xml"
    for name in ("openvino_model.xml", "openvino_language_model.xml"):
        if (d / name).exists():
            return name
    return None


def _vlm_lm_logits(lm, emb_req, input_ids):
    """Run the standalone VLM text decoder on embedded token ids -> logits."""
    import numpy as np
    import torch

    emb_in = emb_req.inputs[0].get_any_name()
    emb_out = emb_req.outputs[0].get_any_name()
    ids = input_ids.cpu().numpy().astype(np.int64)
    embeds = emb_req({emb_in: ids})[emb_out]
    B, S = embeds.shape[0], embeds.shape[1]
    inputs = {"inputs_embeds": embeds}
    if "attention_mask" in lm.input_names:
        inputs["attention_mask"] = np.ones((B, S), dtype=np.int64)
    if "position_ids" in lm.input_names:
        pos = np.arange(S, dtype=np.int64)[None, :]
        mt = getattr(lm.config, "model_type", "")
        if mt in ("qwen3_5", "qwen3_5_moe") and pos.ndim != 3:
            pos = np.repeat(np.expand_dims(pos, 0), 4, axis=0)
        elif mt in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl") and pos.ndim == 2:
            pos = np.repeat(np.expand_dims(pos, 0), 3, axis=0)
        elif mt == "qwen3_omni_moe" and pos.ndim == 2:
            pos = np.repeat(np.expand_dims(pos, 0), 4, axis=0)
        inputs["position_ids"] = pos
    if "beam_idx" in lm.input_names:
        inputs["beam_idx"] = np.arange(B, dtype=np.int64)
    if "token_type_ids" in lm.input_names:
        inputs["token_type_ids"] = np.zeros((B, S), dtype=np.int64)
    req = lm.request
    reset = getattr(req, "reset_state", None)
    if callable(reset):
        reset()
    req.infer(inputs)
    return torch.from_numpy(req.get_tensor("logits").data).clone()


def e2e(model_dir: str | Path, *, log=None) -> dict:
    """End-to-end GenAI smoke on a text OR VLM dir (reuses genai_test.run_test)."""
    from ov_converter.genai_test import run_test
    d = Path(model_dir)
    if log:
        log(f"e2e: running GenAI smoke test on {d.name}")
    return run_test(model_dir, prompt=_cfg.prompt, max_new_tokens=_cfg.max_new_tokens,
                    device=_cfg.device, log=log)


def perplexity(model_dir: str | Path, *, log=None, label: str = "model",
               progress=None) -> dict:
    """Causal-LM perplexity over the eval corpus.

    Text-only dirs use openvino_model.xml. VLM dirs fall back to the standalone
    openvino_language_model.xml decoder, embedded via openvino_text_embeddings_model.xml.
    """
    d = Path(model_dir)
    ir_name = _resolve_language_ir(d)
    if ir_name is None:
        return {
            "ok": False,
            "reason": "perplexity requires a text-only model (openvino_model.xml) "
                      "or a VLM language submodel (openvino_language_model.xml)",
        }
    try:
        import torch
        import torch.nn.functional as F
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        is_lm_submodel = ir_name == "openvino_language_model.xml"
        if log:
            log(f"perplexity: loading {label} {d.name}")
        model = OVModelForCausalLM.from_pretrained(
            str(d), file_name=ir_name, compile=True, trust_remote_code=True
        )
        tok = AutoTokenizer.from_pretrained(str(d))
        emb_req = None
        if is_lm_submodel:
            import numpy as np
            import openvino as ov

            core = ov.Core()
            if log:
                log(f"perplexity: loading text embeddings submodel ({label})")
            emb_req = core.compile_model(str(d / "openvino_text_embeddings_model.xml"), _cfg.device)
        if log:
            log(f"perplexity: model compiled, tokenizer ready ({label})")
        lines = [ln for ln in corpus_lines() if len(ln) >= 20]
        if log:
            log(f"perplexity: processing {len(lines)} lines ({label})")
        total_nll = 0.0
        total_tokens = 0
        done = 0
        with torch.no_grad():
            for ln in lines:
                enc = tok(ln, return_tensors="pt", truncation=True, max_length=512)
                ids = enc["input_ids"]
                if ids.shape[1] < 2:
                    continue
                if is_lm_submodel:
                    logits = _vlm_lm_logits(model, emb_req, ids)
                else:
                    out = model(**enc)
                    logits = out.logits
                shift = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                labels = ids[:, 1:].reshape(-1)
                valid = labels < logits.shape[-1]
                labels = labels.clamp(max=logits.shape[-1] - 1)
                nll = F.cross_entropy(shift, labels, reduction="none") * valid
                total_nll += float(nll.sum())
                total_tokens += int(valid.sum())
                done += 1
                if progress:
                    progress(done / max(len(lines), 1))
                if log and done % 50 == 0:
                    log(f"perplexity: processed {done}/{len(lines)} lines ({label})")
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
            "submodel": ir_name,
            "vocab_size": len(tok),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:300]}


def side_by_side(baseline: str | Path, candidate: str | Path, *, log=None,
                 progress=None) -> dict:
    """Token-level First/Sum Divergent Tokens between baseline and candidate.

    Text-only dirs use openvino_genai.LLMPipeline (greedy). VLM dirs fall back
    to genai_test.run_test outputs. Models are loaded sequentially to limit RAM.
    """
    from transformers import AutoTokenizer

    d_b, d_c = Path(baseline), Path(candidate)
    missing = [str(p) for p in (d_b, d_c) if not p.exists()]
    if missing:
        return {
            "ok": False,
            "status": "na",
            "reason": "side_by_side needs both model dirs; missing: " + ", ".join(missing),
        }
    tok_b = AutoTokenizer.from_pretrained(str(d_b))
    tok_c = AutoTokenizer.from_pretrained(str(d_c))
    if len(tok_b) != len(tok_c):
        return {
            "ok": False,
            "status": "na",
            "reason": (f"baseline and candidate tokenizer vocab sizes differ "
                       f"({len(tok_b)} vs {len(tok_c)})"),
        }
    prompts = _FIXED_PROMPTS[:_cfg.num_prompts] or _FIXED_PROMPTS
    n = len(prompts)
    is_vlm = (d_b / "openvino_vision_embeddings_model.xml").exists() or \
        (d_c / "openvino_vision_embeddings_model.xml").exists()

    if is_vlm:
        from ov_converter.genai_test import run_test
        if log:
            log(f"side_by_side: VLM dirs detected -> genai_test outputs ({n} prompts)")
        base_outs, cand_outs = [], []
        for i, p in enumerate(prompts, 1):
            if log:
                log(f"side_by_side: generating prompt {i}/{n} (baseline)")
            base_outs.append(run_test(baseline, prompt=p, max_new_tokens=_cfg.max_new_tokens,
                                      device=_cfg.device)["output"])
            if progress:
                progress(i / (2 * n))
        for i, p in enumerate(prompts, 1):
            if log:
                log(f"side_by_side: generating prompt {i}/{n} (candidate)")
            cand_outs.append(run_test(candidate, prompt=p, max_new_tokens=_cfg.max_new_tokens,
                                      device=_cfg.device)["output"])
            if progress:
                progress(0.5 + i / (2 * n))
    else:
        import openvino_genai as genai
        if log:
            log(f"side_by_side: compiling baseline LLMPipeline ({_cfg.device}) ...")
        pipe = genai.LLMPipeline(str(d_b), _cfg.device)
        base_outs = []
        for i, p in enumerate(prompts, 1):
            if log:
                log(f"side_by_side: generating prompt {i}/{n} (baseline)")
            base_outs.append(_clean(str(pipe.generate(p, max_new_tokens=_cfg.max_new_tokens, do_sample=False))))
            if progress:
                progress(i / (2 * n))
        del pipe
        gc.collect()
        if log:
            log(f"side_by_side: compiling candidate LLMPipeline ({_cfg.device}) ...")
        pipe = genai.LLMPipeline(str(d_c), _cfg.device)
        cand_outs = []
        for i, p in enumerate(prompts, 1):
            if log:
                log(f"side_by_side: generating prompt {i}/{n} (candidate)")
            cand_outs.append(_clean(str(pipe.generate(p, max_new_tokens=_cfg.max_new_tokens, do_sample=False))))
            if progress:
                progress(0.5 + i / (2 * n))
        del pipe
        gc.collect()

    tok = AutoTokenizer.from_pretrained(str(d_b))
    rows: list[dict] = []
    fdt_vals, sdt_vals, exact_vals = [], [], []
    for p, bo, co in zip(prompts, base_outs, cand_outs):
        bt = tok(bo, add_special_tokens=False)["input_ids"]
        ct = tok(co, add_special_tokens=False)["input_ids"]
        if not bt and not ct:
            fdt_n, sdt_n, exact = 0.0, 0.0, 1
            explain = "Identical output"
        elif not bt:
            fdt_n, sdt_n, exact = 0.0, float(len(ct)), 0
            explain = f"Empty baseline output; SDT {sdt_n} — full divergence"
        else:
            L = len(bt)
            nmin = min(len(bt), len(ct))
            fdt = next((i for i in range(nmin) if bt[i] != ct[i]), nmin)
            sdt = sum(1 for i in range(nmin) if bt[i] != ct[i]) + abs(len(bt) - len(ct))
            exact = 1 if bt == ct else 0
            fdt_n = round(fdt / L, 3)
            sdt_n = round(sdt / L, 3)
            if exact:
                explain = "Identical output"
            else:
                explain = (
                    f"FDT {fdt_n} — first divergence at token {fdt_n * 100:.1f}% of reference; "
                    f"SDT {sdt_n} — ~{sdt_n * 100:.0f}% of tokens differ"
                )
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
            "explain": explain,
        })
    nprompts = max(len(fdt_vals), 1)
    return {
        "prompts": rows,
        "avg_fdt": round(sum(fdt_vals) / nprompts, 3),
        "avg_sdt": round(sum(sdt_vals) / nprompts, 3),
        "exact_match_pct": round(sum(exact_vals) / nprompts * 100, 2),
    }


def _decorate_e2e(res: dict, cand_name: str) -> dict:
    # ok/status convention: ok reflects EXECUTION (True = ran to completion),
    # status the VERDICT. e2e has no "completed but degraded" state, so
    # ok:false == genuine failure (status "bad", rc=1); ok:true == completed
    # (status "ok", rc=0).
    res["label"] = "E2E GenAI smoke"
    if res.get("ok"):
        tp = res.get("tok_per_s")
        res["status"] = "ok"
        res["summary"] = f"E2E ok — {tp} tok/s"
        res["details"] = (
            f"End-to-end GenAI smoke test on {cand_name} "
            f"({'VLMPipeline' if res.get('is_vlm') else 'LLMPipeline'} on {_cfg.device}, "
            f"max_new_tokens {_cfg.max_new_tokens}).\n"
            f"Generated {res.get('tokens')} tokens in {res.get('elapsed_s')} s -> {tp} tok/s.\n"
            f"Prompt: {res.get('prompt')!r}\n"
            f"Output (truncated): {res.get('output')!r}"
        )
    else:
        res["status"] = "bad"
        msg = res.get("error") or res.get("reason") or "see details"
        res["summary"] = f"E2E failed — {msg}"
        res["details"] = f"End-to-end GenAI smoke test on {cand_name} failed.\n{msg}"
    return res


def _decorate_perplexity(res: dict, bname: str, cname: str, corpus_info: dict) -> dict:
    b, c = res.get("baseline", {}), res.get("candidate", {})
    res["label"] = "Perplexity"
    if res.get("status") == "na":
        return res
    b_ok = isinstance(b, dict) and b.get("ok")
    c_ok = isinstance(c, dict) and c.get("ok")
    if b_ok and c_ok and b.get("ppl") is not None and c.get("ppl") is not None:
        pb, pc = b["ppl"], c["ppl"]
        delta = (pc - pb) / pb * 100 if pb else 0.0
        if delta >= 50:
            status, verdict = "bad", "major degradation"
        elif delta >= 10:
            status, verdict = "warn", "noticeable drop"
        elif delta < 0:
            status, verdict = "ok", "improvement (lower is better)"
        else:
            status, verdict = "ok", "stable"
        res["status"] = status
        res["summary"] = f"Perplexity {pb} → {pc} ({delta:+.0f}%) — {verdict}"
        res["details"] = (
            f"Perplexity of the causal-LM over the eval corpus (lower is better).\n"
            f"Baseline {bname}: ppl {pb} over {b.get('lines')} lines / {b.get('tokens')} tokens "
            f"({b.get('submodel')})\n"
            f"Candidate {cname}: ppl {pc} over {c.get('lines')} lines / {c.get('tokens')} tokens "
            f"({c.get('submodel')})\n"
            f"Delta {delta:+.1f}% — {verdict}. Thresholds: green < +10%, amber < +50%, red >= +50%.\n"
            f"Corpus: {corpus_info['file']} ({corpus_info['lines']} lines)"
        )
    else:
        res["status"] = "bad"
        b_msg = (b.get("reason") or b.get("error") or "failed") if isinstance(b, dict) else "failed"
        c_msg = (c.get("reason") or c.get("error") or "failed") if isinstance(c, dict) else "failed"
        if b_ok or c_ok:
            side = "baseline" if not b_ok else "candidate"
            res["summary"] = f"Perplexity unavailable — {side} failed ({b_msg if not b_ok else c_msg})"
        else:
            res["summary"] = "Perplexity unavailable — both sides failed"
        res["details"] = (
            f"Perplexity of the causal-LM over the eval corpus (lower is better).\n"
            f"Baseline {bname}: {'ppl ' + str(b.get('ppl')) if b_ok else 'FAILED — ' + b_msg}\n"
            f"Candidate {cname}: {'ppl ' + str(c.get('ppl')) if c_ok else 'FAILED — ' + c_msg}\n"
            f"Corpus: {corpus_info['file']} ({corpus_info['lines']} lines)"
        )
    return res


def _decorate_sbs(res: dict, bname: str, cname: str) -> dict:
    res["label"] = "Side-by-side generation"
    if not res.get("ok"):
        # Deliberate skip (missing dirs / tokenizer mismatch) is ok:false with
        # status "na"; any other non-ok result (e.g. a raised exception) is a
        # genuine failure and must NOT masquerade as a skip.
        if res.get("status") == "na":
            msg = res.get("reason") or res.get("error") or "see details"
            res["status"] = "na"
            res["summary"] = f"Side-by-side skipped — {msg}"
            res["details"] = f"Token-level comparison of {bname} vs {cname} could not be run.\n{msg}"
            return res
        res["status"] = "bad"
        msg = res.get("error") or res.get("reason") or "see details"
        res["summary"] = f"Side-by-side failed — {msg}"
        res["details"] = f"Token-level comparison of {bname} vs {cname} failed.\n{msg}"
        return res
    ex = res.get("exact_match_pct", 0.0)
    n = len(res.get("prompts", []))
    if ex >= 90:
        status = "ok"
    elif ex >= 50:
        status = "warn"
    else:
        status = "bad"
    res["status"] = status
    band = "green" if status == "ok" else ("amber" if status == "warn" else "red")
    res["summary"] = (
        f"Side-by-side — exact match {ex}% · avg FDT {res.get('avg_fdt')} · avg SDT {res.get('avg_sdt')}"
    )
    res["details"] = (
        f"Token-level comparison of greedy generation across {n} prompts ({bname} vs {cname}).\n"
        f"Exact match: {ex}% — {band}.\n"
        f"Avg FDT (first divergent token, fraction of reference): {res.get('avg_fdt')}\n"
        f"Avg SDT (sum of divergent tokens, fraction of reference): {res.get('avg_sdt')}\n"
        f"Per-prompt details are in prompts[].explain."
    )
    return res


def _decorate_test(name: str, res: dict, *, baseline_name: str,
                   candidate_name: str, corpus_info: dict) -> dict:
    if name == "e2e":
        return _decorate_e2e(res, candidate_name)
    if name == "perplexity":
        return _decorate_perplexity(res, baseline_name, candidate_name, corpus_info)
    if name == "side_by_side":
        return _decorate_sbs(res, baseline_name, candidate_name)
    return res


def _done_detail(name: str, res: dict) -> str:
    if name == "e2e":
        return f"{res.get('tok_per_s')} tok/s"
    if name == "perplexity":
        b = res.get("baseline") if isinstance(res.get("baseline"), dict) else None
        sub = res.get("candidate") if isinstance(res.get("candidate"), dict) else res
        pb = b.get("ppl") if b is not None else None
        if pb is not None:
            return f"ppl={pb} → {sub.get('ppl')} over {sub.get('lines')} lines"
        return f"ppl={sub.get('ppl')} over {sub.get('lines')} lines"
    if name == "side_by_side":
        return f"avg_fdt={res.get('avg_fdt')} exact={res.get('exact_match_pct')}%"
    return "ok"


def _comparable_perplexity(b: dict, c: dict) -> str | None:
    if b.get("submodel") != c.get("submodel"):
        return (f"baseline and candidate are not the same architecture/tokenizer "
                f"(submodel: {b.get('submodel')} vs {c.get('submodel')})")
    if b.get("vocab_size") != c.get("vocab_size"):
        return (f"baseline and candidate are not the same architecture/tokenizer "
                f"(vocab size: {b.get('vocab_size')} vs {c.get('vocab_size')})")
    return None


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
            emit.done("testing", f"rc=1")
            emit.emit("META", "done", json.dumps(result, ensure_ascii=False, default=str))
            return 1

    emit.start("corpus")
    corpus_file = str(Path(_cfg.corpus) if _cfg.corpus else DEFAULT_CORPUS)
    corpus_info = {"file": corpus_file, "lines": 0}
    try:
        lines = corpus_lines()
        corpus_info["lines"] = len(lines)
        result["corpus"] = {"file": corpus_file, "lines": len(lines)}
        emit.done("corpus", f"{len(lines)} lines")
    except Exception as e:  # noqa: BLE001
        result["corpus"] = {"ok": False, "error": str(e)[:300]}
        emit.fail("corpus", str(e)[:300])
        rc = 1

    n_tests = len(_cfg.tests)

    def _report(idx: int, portion: float) -> None:
        emit.progress(int(round((idx + portion) / max(n_tests, 1) * 100)))

    baseline_name = Path(_cfg.baseline).name if _cfg.baseline else ""
    candidate_name = Path(_cfg.candidate).name if _cfg.candidate else ""

    # ok/status convention for every sub-test result:
    #   ok     = the stage EXECUTED (ran to completion without exception/skip)
    #   status = quality verdict: "ok"/"warn"/"bad" (ok stays True for warn/bad —
    #            quality degradation must NOT fail the task, rc stays 0),
    #            or "na" for a deliberate skip (ok False, rc unaffected).
    # A non-ok, non-"na" result is a genuine failure -> emit fail + rc = 1.
    for i, name in enumerate(_cfg.tests):
        emit.start(name)
        res: dict | None = None
        try:
            if name == "e2e":
                res = e2e(_cfg.candidate, log=log)
                _report(i, 1.0)
            elif name == "perplexity":
                b = perplexity(_cfg.baseline, log=log, label="baseline",
                               progress=lambda p: _report(i, p * 0.5))
                c = perplexity(_cfg.candidate, log=log, label="candidate",
                               progress=lambda p: _report(i, 0.5 + p * 0.5))
                res = {"baseline": b, "candidate": c}
                if b.get("ok") and c.get("ok"):
                    reason = _comparable_perplexity(b, c)
                    if reason:
                        res.update({
                            "ok": False,
                            "status": "na",
                            "label": "Perplexity",
                            "reason": reason,
                            "summary": f"Perplexity skipped — {reason}",
                            "details": (f"Perplexity of the causal-LM over the eval corpus "
                                        f"(lower is better).\n{reason}"),
                        })
                    else:
                        res["ok"] = True
                else:
                    res["ok"] = False
                _report(i, 1.0)
            elif name == "side_by_side":
                res = side_by_side(_cfg.baseline, _cfg.candidate, log=log,
                                   progress=lambda p: _report(i, p))
                res.setdefault("ok", True)
            else:
                raise ValueError(f"unknown test: {name}")
        except Exception as e:  # noqa: BLE001
            # genuine failure; the single terminal emit happens below, never here
            res = {"ok": False, "error": str(e)[:300]}

        try:
            res = _decorate_test(
                name, res,
                baseline_name=baseline_name,
                candidate_name=candidate_name,
                corpus_info=corpus_info,
            )
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "status": "bad", "error": str(e)[:300]}
        if not isinstance(res, dict):
            res = {"ok": False, "status": "bad", "error": f"non-dict result: {res!r}"}

        result["tests"][name] = res
        status = res.get("status")
        if status == "na":
            msg = res.get("summary") or res.get("reason") or "skipped"
            emit.done(name, f"na {str(msg)[:200]}")
        elif res.get("ok"):
            emit.done(name, _done_detail(name, res))
        else:
            msg = res.get("summary") or res.get("reason") or res.get("error") or "failed"
            emit.fail(name, str(msg)[:200])
            rc = 1

    emit.done("testing", f"rc={rc}")
    emit.emit("META", "done", json.dumps(result, ensure_ascii=False, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())