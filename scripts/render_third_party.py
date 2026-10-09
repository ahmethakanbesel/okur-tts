"""Render third-party Turkish TTS systems in isolated environments (no project imports). Run from the repo root with -I:

    uv run --no-project --with piper-tts --with soundfile python -I scripts/render_third_party.py \
        piper runs/third_party/piper/dfki/tr_TR-dfki-medium.onnx <sentences.json> <out_dir>
    uv run --no-project --with "transformers<5" --with torch --with soundfile python -I scripts/render_third_party.py \
        mms runs/third_party/mms_tur <sentences.json> <out_dir>

<sentences.json> is a list of {id, text}. Writes <out_dir>/<id>.wav and <out_dir>/timing.json (audio seconds and
compute seconds per sentence, model load and one warm-up excluded).
"""

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

Synth = Callable[[str], tuple[np.ndarray, int]]


def piper(model: str) -> tuple[Synth, str]:
    from piper import PiperVoice

    voice = PiperVoice.load(model)

    def say(text: str) -> tuple[np.ndarray, int]:
        chunks = list(voice.synthesize(text))  # one chunk per sentence; espeak-ng phonemizes
        return np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate

    return say, "ONNX Runtime"


def mms(model_dir: str) -> tuple[Synth, str]:
    import torch
    from transformers import AutoTokenizer, VitsModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = VitsModel.from_pretrained(model_dir).eval()

    @torch.no_grad()
    def say(text: str) -> tuple[np.ndarray, int]:
        torch.manual_seed(0)  # VITS samples noise: fixed seed per sentence
        # The tokenizer lowercases with str.lower(), which maps I to i; Turkish needs I -> ı, İ -> i.
        inputs = tokenizer(text.replace("I", "ı").replace("İ", "i"), return_tensors="pt")
        return model(**inputs).waveform[0].numpy(), model.config.sampling_rate

    return say, "PyTorch"


def main(kind: str, model: str, sentences_path: str, out: str) -> None:
    say, library = {"piper": piper, "mms": mms}[kind](model)
    threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    device = f"CPU, {threads} threads ({library})"
    sentences = json.loads(Path(sentences_path).read_text(encoding="utf-8"))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    say(sentences[0]["text"])  # warm-up
    timing = {}
    for s in sentences:
        t0 = time.perf_counter()
        audio, rate = say(s["text"])
        compute = time.perf_counter() - t0
        sf.write(out_dir / f"{s['id']}.wav", audio.astype(np.float32), rate)
        timing[s["id"]] = {"audio_s": len(audio) / rate, "compute_s": compute}
    (out_dir / "timing.json").write_text(json.dumps({"device": device, "per_sentence": timing}, indent=1))
    total_audio = sum(t["audio_s"] for t in timing.values())
    total_compute = sum(t["compute_s"] for t in timing.values())
    print(f"{kind} {model}: {len(timing)} sentences, RTF(audio/compute) {total_audio / total_compute:.1f}x on {device}")


if __name__ == "__main__":
    main(*sys.argv[1:])
