"""Intelligibility: synthesize sentences, transcribe them with the Turkish CTC recognizer, report character error rate.

A cheap stand-in for Whisper-large-v3 WER that runs on a Mac during training. The final comparison with EMA Lightning
uses Whisper on Freya-TR-Eval; this one tracks progress.
"""

import numpy as np

from okur.data.align import Recognizer
from okur.data.audio import resample
from okur.data.codec import RATE

SENTENCES = [
    "Bugün hava çok güzel, parkta yürüyüş yapmaya karar verdik.",
    "Toplantı yarın sabah saat dokuzda başlayacak.",
    "Bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor.",
    "Ofisimiz üçüncü katta, asansörün hemen yanında.",
    "Çocuklar bahçede top oynarken annesi onları izliyordu.",
    "Yeni kütüphane şehrin merkezinde açıldı.",
]


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _plain(text: str) -> str:
    return " ".join("".join(c for c in text.translate(str.maketrans("âîû", "aiu")) if c.isalpha() or c == " ").split())


def greedy(recognizer: Recognizer, audio48: np.ndarray) -> str:
    logp = recognizer.emissions([resample(audio48.astype(np.float32), RATE, 16000)])[0]
    inv = {v: k for k, v in recognizer.vocab.items()}
    out, prev = [], -1
    for x in logp.argmax(-1):
        if x not in (prev, recognizer.blank):
            out.append(inv[int(x)])
        prev = x
    return "".join(out).replace("|", " ")


def character_error_rate(pairs: list[tuple[str, str]]) -> float:
    """pairs of (reference letters, hypothesis); both reduced to plain lowercase letters and spaces."""
    errors = sum(_edit_distance(_plain(ref), _plain(hyp)) for ref, hyp in pairs)
    return errors / max(1, sum(len(_plain(ref)) for ref, _ in pairs))
