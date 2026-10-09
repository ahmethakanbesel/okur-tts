"""Freya-TR-Eval: our model vs EMA Lightning, scored identically.

Protocol (EMA Lightning's model card): all 495 sentences, raw text in, 3 seeds, audio band-limited to 8 kHz (16 kHz
sampling), Whisper-large-v3 with beam 5, then WER and CER after the same text normalization for every system.

Contamination: Freya's "short-native" sentences were sampled from Common Voice 17 texts, and we train on Common
Voice 27. Every sentence that appears in our training transcripts is flagged, and scores are reported on all 495
and on the clean (never-seen) subset.
"""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from okur.data.audio import resample
from okur.frontend.alphabet import turkish_lower

log = logging.getLogger(__name__)

REPO = "freyavoice/freya-tr-eval"
WHISPER = "openai/whisper-large-v3"
SEEDS = (0, 1, 2)


def load_sentences() -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "freya_tr_eval.jsonl", repo_type="dataset")
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    """Lowercase the Turkish way, fold circumflexes (Whisper writes them inconsistently), drop punctuation."""
    text = turkish_lower(text).translate(str.maketrans("âîû", "aiu"))
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def error_rates(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """(WER, CER) over (reference, hypothesis) pairs, both normalized."""
    refs = [normalize(r) for r, _ in pairs]
    hyps = [normalize(h) for _, h in pairs]
    pairs_n = list(zip(refs, hyps, strict=True))
    wer = sum(_distance(r.split(), h.split()) for r, h in pairs_n) / sum(len(r.split()) for r in refs)
    cer = sum(_distance(list(r), list(h)) for r, h in pairs_n) / sum(len(r) for r in refs)
    return wer, cer


def training_texts(shard_dirs: list[Path]) -> set[str]:
    import pyarrow.parquet as pq

    seen: set[str] = set()
    for d in shard_dirs:
        for p in sorted(d.glob("shard-*.parquet")):
            seen.update(normalize(t) for t in pq.read_table(p, columns=["text"]).column("text").to_pylist())
    return seen


class Whisper:
    def __init__(self, device: str, batch: int = 16) -> None:
        from transformers import pipeline

        self.batch = batch
        self.asr: Any = pipeline("automatic-speech-recognition", model=WHISPER, torch_dtype=torch.float16,
                                 device=device)

    def __call__(self, audio16: list[np.ndarray]) -> list[str]:
        out = self.asr([{"raw": a, "sampling_rate": 16000} for a in audio16], batch_size=self.batch,
                       generate_kwargs={"language": "turkish", "task": "transcribe", "num_beams": 5})
        return [o["text"].strip() for o in out]


Synth = Callable[[str, int], tuple[np.ndarray, int]]  # (text, seed) -> (audio, sample rate)


def evaluate(name: str, synth: Synth, sentences: list[dict[str, Any]], whisper: Whisper, seen: set[str],
             out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        audio16 = []
        for s in sentences:
            audio, rate = synth(s["text"], seed)
            audio16.append(resample(audio.astype(np.float32), rate, 16000))
        for s, hyp in zip(sentences, whisper(audio16), strict=True):
            rows.append({"id": s["id"], "seed": seed, "ref": s["text"], "hyp": hyp,
                         "seen_in_training": normalize(s["text"]) in seen})
        log.info("%s: seed %d done", name, seed)
    (out_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    result: dict[str, Any] = {"system": name}
    subsets: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
        ("all", lambda _: True), ("clean", lambda r: not r["seen_in_training"]))
    for subset, keep in subsets:
        per_seed = [error_rates([(r["ref"], r["hyp"]) for r in rows if r["seed"] == seed and keep(r)])
                    for seed in SEEDS]
        wers, cers = [w for w, _ in per_seed], [c for _, c in per_seed]
        result[subset] = {"wer": float(np.mean(wers)), "wer_spread": float(np.max(wers) - np.min(wers)),
                          "cer": float(np.mean(cers)), "n": sum(1 for r in rows if r["seed"] == 0 and keep(r))}
    return result


def run(checkpoint: Path, shard_dirs: list[Path], out_dir: Path, *, decoder: Path | None, device: str = "cuda",
        steps: int = 16, guidance: float = 3.0, with_ema: bool = True, speaker: str | None = None,
        whisper_batch: int = 16) -> None:
    from okur.data.codec import RATE, Codec
    from okur.model.acoustic import Acoustic
    from okur.model.config import AcousticConfig, DecoderConfig
    from okur.model.decoder import Decoder
    from okur.synth import Synthesizer

    out_dir.mkdir(parents=True, exist_ok=True)
    sentences = load_sentences()
    seen = training_texts(shard_dirs)
    log.info("%d sentences, %d seen in training transcripts", len(sentences),
             sum(normalize(s["text"]) in seen for s in sentences))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Acoustic(AcousticConfig.model_validate(state["model_config"]))
    model.load_state_dict(state["average"])
    model.to(device)
    if decoder is not None:
        dstate = torch.load(decoder, map_location="cpu", weights_only=False)
        dec = Decoder(DecoderConfig.model_validate(dstate["model_config"])).to(device).eval()
        dec.load_state_dict(dstate["model"])

        @torch.no_grad()
        def decode(z: torch.Tensor) -> np.ndarray:
            return dec(z.T[None].float())[0].float().cpu().numpy()
    else:
        codec = Codec(device)
        decode = codec.decode  # type: ignore[assignment]
    synth = Synthesizer(model, decode, device)
    spk = state["speakers"].index(speaker) if speaker else 0

    def ours(text: str, seed: int) -> tuple[np.ndarray, int]:
        return synth(text, speaker=spk, steps=steps, guidance=guidance, seed=seed), RATE

    whisper = Whisper(device, whisper_batch)
    results = [evaluate("ours" + ("" if decoder else "_ace_decoder"), ours, sentences, whisper, seen, out_dir)]
    if with_ema:
        from ema_lightning import EMA

        ema = EMA(device=device)

        def ema_say(text: str, seed: int) -> tuple[np.ndarray, int]:
            speech: Any = ema.say(text, seed=seed)  # one text in, one Speech out
            return speech.audio, speech.sample_rate

        results.append(evaluate("ema_lightning", ema_say, sentences, whisper, seen, out_dir))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\n{'system':24s} {'WER all':>9s} {'CER all':>9s} {'WER clean':>10s} {'CER clean':>10s}  n clean")
    for r in results:
        print(f"{r['system']:24s} {100 * r['all']['wer']:8.2f}% {100 * r['all']['cer']:8.2f}% "
              f"{100 * r['clean']['wer']:9.2f}% {100 * r['clean']['cer']:9.2f}%  {r['clean']['n']}")
