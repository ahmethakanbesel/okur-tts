"""Reference outputs of the web pipeline (okur.export_web.OnnxSynthesizer) for the TypeScript tests.

    uv run python scripts/web_fixtures.py web/public/models web/test/fixtures.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from okur.export_web import OnnxSynthesizer, gaussian_noise, timeline
from okur.frontend import encode

TEXTS = [
    "Bu konuyla alakalı olarak şirketin karı hala artıyor.",
    "Toplantı 3. katta, 14 Mart 2026 saat 09:30'da; fiyat 1.250,75 TL.",
    "Yarın sabah erkenden yola çıkacak mıyız?",
]


def main(models: Path, out: Path) -> None:
    synth = OnnxSynthesizer(models)
    cases = []
    for text in TEXTS:
        letters = synth.frontend(text)
        ids = np.array([encode(letters)], dtype=np.int64)
        _, dur = synth.sessions["text"].run(None, {"ids": ids})
        cw, wstart, fw, fp = timeline(letters, dur[0])
        audio = synth.say(text, seed=0)
        steps, c = len(synth.config["times"]), synth.config["latent_dim"]
        cases.append({
            "text": text, "letters": letters, "dur": dur[0].tolist(), "cw": cw.tolist(), "wstart": wstart.tolist(),
            "fw": fw.tolist(), "fp": fp.tolist(), "noise_head": gaussian_noise(0, steps * len(fw) * c)[:64].tolist(),
            "audio_length": len(audio), "audio_rms": float(np.sqrt(np.mean(audio**2))),
            "audio_head": audio[:4800].tolist(),
        })
    out.write_text(json.dumps({"cases": cases}))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
