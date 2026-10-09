"""Render the demo's example sentences with FreyaTTS-small (freyavoice/Freya-TTS, Apache-2.0) through its official
pipeline (its own text normalization, clause chunking and voicing retry; 32 Euler steps; the canonical voice seed).

    git clone https://github.com/freyavoiceai/FreyaTTS <src>
    uv run --no-project --python 3.12 --with-requirements <src>/requirements.txt python -I scripts/render_freya.py \
        <src> demo/hard_sentences.json runs/hard/freya_small

Writes <out>/<id>.wav (48 kHz) and <out>/timing.json.
"""

import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch


def main(src: str, sentences_path: str, out: str) -> None:
    sys.path.insert(0, str(Path(src).resolve()))
    from freyatts import FreyaTTS  # the official package, from the cloned repository

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tts = FreyaTTS.from_pretrained("freyavoice/freya-tts", device=device)
    sentences = json.loads(Path(sentences_path).read_text(encoding="utf-8"))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tts.synthesize(sentences[0]["text"])  # warm-up
    timing = {}
    for s in sentences:
        t0 = time.perf_counter()
        wav = tts.synthesize(s["text"])  # default steps (32) and the canonical voice seed
        timing[s["id"]] = {"audio_s": len(wav) / tts.sample_rate, "compute_s": time.perf_counter() - t0}
        sf.write(out_dir / f"{s['id']}.wav", wav, tts.sample_rate)
        print(s["id"], f"{timing[s['id']]['audio_s']:.1f} s audio", flush=True)
    (out_dir / "timing.json").write_text(json.dumps({"device": device, "per_sentence": timing}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
