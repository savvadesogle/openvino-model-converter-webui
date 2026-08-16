"""End-to-end sanity check of a converted model with OpenVINO GenAI."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import ov_converter.settings as S


def _sample_image(size: int = 224) -> "np.ndarray":
    from PIL import Image
    import numpy as np

    img = Image.new("RGB", (size, size), (180, 220, 200))
    for i in range(0, size, 16):
        for j in range(0, size, 16):
            if (i + j) % 32 == 0:
                for dx in range(8):
                    for dy in range(8):
                        img.putpixel((i + dx, j + dy), (220, 180, 200))
    return np.asarray(img)  # uint8 HxWxC


def run_test(model_dir: str | Path, *, prompt: str | None = None,
             max_new_tokens: int = 24, device: str = "CPU",
             log: Callable[[str], None] | None = None) -> dict:
    """Load the model dir with GenAI, run a short generation, return diagnostics."""
    import openvino_genai as genai

    d = Path(model_dir)
    is_vlm = (d / "openvino_vision_embeddings_model.xml").exists()
    text_prompt = prompt or ("Hello, describe this image in one sentence."
                             if is_vlm else "Hello, my name is")

    if log:
        log(f"GenAI pipeline: {'VLMPipeline' if is_vlm else 'LLMPipeline'} ({device})")
    t0 = time.time()
    if is_vlm:
        import openvino as ov
        pipe = genai.VLMPipeline(str(d), device)
        images = [ov.Tensor(_sample_image())]
        result = pipe.generate(text_prompt, images=images, max_new_tokens=max_new_tokens)
    else:
        pipe = genai.LLMPipeline(str(d), device)
        result = pipe.generate(text_prompt, max_new_tokens=max_new_tokens)
    dt = time.time() - t0
    tok = max(result.get_num_generated_tokens(), 1) if hasattr(result, "get_num_generated_tokens") else max_new_tokens
    out = str(result).encode("utf-8", errors="replace").decode("utf-8")

    return {
        "ok": True,
        "is_vlm": is_vlm,
        "prompt": text_prompt,
        "output": out[:400],
        "tokens": tok,
        "elapsed_s": round(dt, 2),
        "tok_per_s": round(tok / max(dt, 0.001), 2),
    }


def format_result(res: dict) -> str:
    head = f"GenAI test: PASS ({res['tokens']} tok, {res['tok_per_s']} tok/s)"
    return head + "\n--- output ---\n" + res["output"]
