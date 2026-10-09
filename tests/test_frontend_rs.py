"""The Rust frontend (browser, mobile) must read text exactly like the Python one (training, server)."""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from okur.evals.freya import load_sentences
from okur.frontend import Frontend

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "frontend-rs"
pytestmark = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs a Rust toolchain")


def _synthetic(n: int, seed: int = 0) -> list[str]:
    """Number-heavy and messy text: ordinals, dates, money, times, ranges, punctuation, typography, bidi, decomposed
    accents."""
    rng = random.Random(seed)
    words = ["kat", "Dünya", "savaşı", "şirketin", "karı", "hala", "alakalı", "Mart", "TL", "kg", "saat", "ve",
             "İstanbul",
             "IŞIK", "kar", "yağdı", "dükkanın", "mekanik", "%", "€", "$", "Dr.", "vb.", "km", "A.Ş.", "NATO", "café",
             "naïve", "“tırnak”", "—", "…", "\u202e", "â", "kâr", "hâlâ", "😀", "\t", "\n", "1.", "2.", "3.",
             "15.10.2026", "1.250,75", "09:30", "3-5", "%12,5'lik", "2025'te", "1990'larda", "0532 123 45 67", "IV.",
             "XIX.", "25 TL", "5 kg", "12.", "100.", "1.000.000", "3,14", "½", "x²", "_", "a_b", "١٢"]
    out = []
    for _ in range(n):
        k = rng.randint(1, 12)
        seps = [" ", " ", " ", ", ", ". ", "; ", "! ", "? ", "\t", ""]
        out.append("".join(rng.choice(words) + rng.choice(seps) for _ in range(k)))
    return out


def _texts() -> list[str]:
    texts = [s["text"] for s in load_sentences()]
    texts += [s["text"] for s in json.loads((ROOT / "demo/hard_sentences.json").read_text())]
    for shard in sorted((ROOT / "data/prepared/mediaspeech_tr").glob("shard-*.parquet"))[:4]:
        texts += pq.read_table(shard, columns=["text"]).column("text").to_pylist()
    return texts + _synthetic(3000)


def test_rust_frontend_matches_python() -> None:
    build = subprocess.run(["cargo", "build", "--release", "--features", "cli", "-q"], cwd=CRATE, capture_output=True,
                           text=True, check=False)
    assert build.returncode == 0, build.stderr
    texts = _texts()
    run = subprocess.run([str(CRATE / "target/release/okur-frontend")],
                         input="".join(json.dumps(t) + "\n" for t in texts), capture_output=True, text=True, check=True)
    rust = [json.loads(line) for line in run.stdout.splitlines()]
    frontend = Frontend()
    mismatches = [(t, p, r) for t, r in zip(texts, rust, strict=True) if (p := frontend(t)) != r]
    shown = "\n".join(f"{t!r}\n  py: {p!r}\n  rs: {r!r}" for t, p, r in mismatches[:10])
    assert not mismatches, f"{len(mismatches)}/{len(texts)} differ:\n{shown}"
