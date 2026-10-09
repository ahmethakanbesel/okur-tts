"""Compare our Turkish TTS with other open Turkish systems for the web demo. scripts/compare_all.sh runs the whole
sequence on a CUDA machine:

    compare_systems.py export      # Freya sentences -> runs/compare/freya_sentences.json
    compare_systems.py render      # ours (decoder 70k and 110k) + EMA Lightning on GPU -> runs/hard/, runs/utmos/
    render_third_party.py ...      # Piper, MMS (isolated envs) -> runs/hard/, runs/utmos/
    compare_systems.py bench ours|ema_lightning   # CPU speed on the hard sentences -> runs/bench/<system>/timing.json
    compare_systems.py asr hard|freya <systems>   # Whisper-large-v3 transcripts -> runs/compare/asr/
    utmos_score.py runs/utmos ; utmos_score.py runs/hard
    compare_systems.py mp3         # demo/audio/<system>/<id>.mp3, loudness-normalized
    compare_systems.py metrics     # demo/metrics.json, demo/samples.json

Speed runs every system on the same CPU cores (the caller pins them with taskset); ours runs the ONNX graphs the
browser ships (okur.export_web).

Freya WER/CER here use one seed (0) for every system, so they are comparable with each other but not with the
3-seed GPU numbers. "clean" = sentences not in our training transcripts; the flags are taken from the GPU eval's
per-row output (same `training_texts` definition), since the training shards are not on this machine.
"""

import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

COMPARE = Path("runs/compare")
HARD = Path("runs/hard")
BENCH = Path("runs/bench")
WEB_EXPORT = Path("runs/web_export/fp16")
UTMOS = Path("runs/utmos")
DEMO = Path("demo")
FREYA_JSON = COMPARE / "freya_sentences.json"
SEEN_FROM = next(p for p in (Path("runs/eval_freya/v02-student-dec110k/ours.jsonl"),
                             Path("runs/gpu/eval_freya/v02-student-dec110k/ours.jsonl")) if p.exists())
VOICE = "cv:ca179eb54e4e"
ACOUSTIC = "models/v0.2/acoustic_student4_v2c_ca179eb54e4e.pt"
DECODER = "models/v0.2/decoder_3m_step110000.pt"
DECODER_70K = "models/v0.2/decoder_3m_step70000.pt"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
DEVICE = "cuda"
DEMO_SYSTEMS = ("ours", "ema_lightning", "freya_small", "piper_dfki", "mms_tur")  # freya: scripts/render_freya.py
FREYA_DIR = {"ours": "v02_dec110k"}  # demo system -> its Freya render directory under runs/utmos

SYSTEMS = {
    "ours": {"label": "Ours (v0.2: 4-step student + 3M decoder, 110k steps)", "model_url": None,
             "license": "Apache-2.0 (project license)", "params": "9.8M (6.8M acoustic + 3.0M decoder)"},
    "v02": {"label": "Ours (v0.2: 4-step student + 3M decoder, 70k steps)", "model_url": None,
            "license": "Apache-2.0 (project license)", "params": "9.8M (6.8M acoustic + 3.0M decoder)"},
    "ema_lightning": {"label": "EMA Lightning", "model_url": "https://huggingface.co/canberkkkkkk/ema-lightning",
                      "license": "Apache-2.0", "params": "8.6M (5.6M DiT + 3M decoder)"},
    "piper_dfki": {"label": "Piper tr_TR-dfki-medium",
                   "model_url": "https://huggingface.co/rhasspy/piper-voices/tree/main/tr/tr_TR/dfki/medium",
                   "license": "CC-BY-NC-SA-4.0 (dataset: DFKI MaryTTS dfki-ot-data); Piper code MIT/GPL",
                   "params": None},
    "mms_tur": {"label": "Meta MMS-TTS Turkish", "model_url": "https://huggingface.co/facebook/mms-tts-tur",
                "license": "CC-BY-NC-4.0", "params": "36.3M (29.0M used at inference; VITS)"},
}


def load_hard() -> list[dict]:
    return json.loads((DEMO / "hard_sentences.json").read_text(encoding="utf-8"))


def export() -> None:
    from okur.evals.freya import load_sentences

    COMPARE.mkdir(parents=True, exist_ok=True)
    rows = [{"id": s["id"], "text": s["text"]} for s in load_sentences()]
    FREYA_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    print(len(rows), "sentences ->", FREYA_JSON)


def render_all(name: str, say, root: Path, sentences: list[dict]) -> None:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    for s in sentences:
        audio, rate = say(s["text"])
        sf.write(out / f"{s['id']}.wav", audio, rate)
    print(name, "->", out, flush=True)


def ours_torch(decoder: str):
    import torch

    from okur.model.acoustic import Acoustic
    from okur.model.config import AcousticConfig, DecoderConfig
    from okur.model.decoder import Decoder
    from okur.synth import Synthesizer

    state = torch.load(ACOUSTIC, map_location="cpu", weights_only=False)
    model = Acoustic(AcousticConfig.model_validate(state["model_config"]))
    model.load_state_dict(state["average"])
    dstate = torch.load(decoder, map_location="cpu", weights_only=False)
    dec = Decoder(DecoderConfig.model_validate(dstate["model_config"])).to(DEVICE).eval()
    dec.load_state_dict(dstate["model"])

    @torch.no_grad()
    def decode(z: torch.Tensor) -> np.ndarray:
        return dec(z.T[None].float())[0].float().cpu().numpy()

    synth = Synthesizer(model.to(DEVICE), decode, DEVICE)
    spk = state["speakers"].index(VOICE)
    return lambda text: (synth(text, speaker=spk, seed=0), 48000)


def ema(device: str):
    from ema_lightning import EMA

    tts = EMA(device=device)

    def say(text: str):
        speech = tts.say(text, seed=0)
        return np.asarray(speech.audio, dtype=np.float32), speech.sample_rate

    return say


def render() -> None:
    hard, freya = load_hard(), json.loads(FREYA_JSON.read_text(encoding="utf-8"))
    say = ours_torch(DECODER)
    render_all("ours", say, HARD, hard)
    render_all("v02_dec110k", say, UTMOS, freya)
    render_all("v02", ours_torch(DECODER_70K), UTMOS, freya)
    say = ema(DEVICE)
    render_all("ema_lightning", say, HARD, hard)
    render_all("ema_lightning", say, UTMOS, freya)


def bench(system: str) -> None:
    """Seconds of audio per second of compute on the CPU cores this process may use, over the 16 hard sentences."""
    import torch

    threads = len(os.sched_getaffinity(0))
    torch.set_num_threads(threads)
    if system == "ours":
        from okur.export_web import OnnxSynthesizer

        synth = OnnxSynthesizer(WEB_EXPORT, threads=threads)
        say = lambda text: (synth.say(text, seed=0), 48000)  # noqa: E731
        device = f"CPU, {threads} threads (ONNX Runtime, fp16 web graphs)"
    elif system == "ours_torch":
        from okur import release
        from okur.synth import torch_synthesizer

        synth = torch_synthesizer(Path(ACOUSTIC), Path(DECODER), "cpu")
        index = release.read(Path(ACOUSTIC))["speakers"].index(VOICE)
        say = lambda text: (synth(text, speaker=index, seed=0), 48000)  # noqa: E731
        device = f"CPU, {threads} threads (PyTorch)"
    else:
        say = ema("cpu")
        device = f"CPU, {threads} threads (PyTorch)"
    out = BENCH / system
    out.mkdir(parents=True, exist_ok=True)
    sentences = load_hard()
    say(sentences[0]["text"])  # warm-up
    timing = {}
    for s in sentences:
        t0 = time.perf_counter()
        audio, rate = say(s["text"])
        timing[s["id"]] = {"audio_s": len(audio) / rate, "compute_s": time.perf_counter() - t0}
        sf.write(out / f"{s['id']}.wav", audio, rate)
    (out / "timing.json").write_text(json.dumps({"device": device, "cpu": cpu_name(), "per_sentence": timing},
                                                indent=1))
    print(system, "bench:", rtf_of(timing), "x real time on", device, flush=True)


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def rtf_of(timing: dict) -> float:
    return sum(v["audio_s"] for v in timing.values()) / sum(v["compute_s"] for v in timing.values())


def seen_flags() -> dict[str, bool]:
    rows = [json.loads(line) for line in SEEN_FROM.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r["seen_in_training"] for r in rows if r["seed"] == 0}


def asr(subset: str, systems: list[str]) -> None:
    import torch

    from okur.data.audio import resample
    from okur.evals.freya import Whisper

    sentences = load_hard() if subset == "hard" else json.loads(FREYA_JSON.read_text(encoding="utf-8"))
    seen = seen_flags() if subset == "freya" else {}
    root = HARD if subset == "hard" else UTMOS
    out = COMPARE / "asr" / subset
    out.mkdir(parents=True, exist_ok=True)
    device = DEVICE if torch.cuda.is_available() else "cpu"
    whisper = Whisper(device, batch=16)
    for system in systems:
        audio16 = []
        for s in sentences:
            audio, rate = sf.read(root / system / f"{s['id']}.wav", dtype="float32")
            audio16.append(resample(audio, rate, 16000))
        hyps = []
        for i in range(0, len(audio16), 64):  # chunks keep memory flat and show progress
            hyps += whisper(audio16[i:i + 64])
        rows = [{"id": s["id"], "ref": s["text"], "hyp": h, "seen_in_training": seen.get(s["id"], False)}
                for s, h in zip(sentences, hyps, strict=True)]
        (out / f"{system}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                                             encoding="utf-8")
        print(subset, system, "transcribed on", device, flush=True)


def mp3() -> None:
    for system in DEMO_SYSTEMS:
        out = DEMO / "audio" / system
        out.mkdir(parents=True, exist_ok=True)
        for s in load_hard():
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(HARD / system / f"{s['id']}.wav"),
                            "-af", "loudnorm=I=-20:TP=-1.5:LRA=11", "-ac", "1", "-ar", "48000",
                            "-c:a", "libmp3lame", "-b:a", "96k", str(out / f"{s['id']}.mp3")], check=True)
        print(system, "mp3 done")


def error_rates_for(path: Path) -> dict[str, float]:
    from okur.evals.freya import error_rates

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wer, cer = error_rates([(r["ref"], r["hyp"]) for r in rows])
    clean = [(r["ref"], r["hyp"]) for r in rows if not r["seen_in_training"]]
    wer_c, cer_c = error_rates(clean)
    return {"wer_all": wer, "cer_all": cer, "wer_clean": wer_c, "cer_clean": cer_c, "n_all": len(rows),
            "n_clean": len(clean)}


def rtf(system: str) -> tuple[float | None, str | None]:
    path = BENCH / system / "timing.json"
    if not path.exists():
        return None, None
    t = json.loads(path.read_text())
    return rtf_of(t["per_sentence"]), t["device"]


def paired(a: dict[str, float], b: dict[str, float]) -> dict:
    ids = sorted(set(a) & set(b))
    d = np.array([b[i] - a[i] for i in ids])
    sd = float(d.std(ddof=1))
    half = 1.96 * sd / math.sqrt(len(d))
    return {"mean_diff": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)], "sd": sd,
            "n": len(d), "fraction_b_better": float((d > 0).mean())}


UNIT_WORDS = ("tl", "kuruş", "lira", "yüzde", "dolar", "avro", "euro")  # a number's unit as a transcript writes it


def _word(w: str) -> str:
    """One comparable token per written word: Turkish lowercase, circumflexes folded, punctuation removed (so
    İstanbul'dan stays one word)."""
    from okur.evals.freya import normalize

    return normalize(w).replace(" ", "")


def heard(system: str, sentence_id: str) -> list[dict]:
    """Whisper's transcript of one hard sentence as words, each marked ok or not against the reference (aligned with
    difflib). Words with digits are not judged: transcripts write numbers in many equivalent ways (9.30 / 09:30)."""
    import difflib

    path = COMPARE / "asr" / "hard" / f"{system}.jsonl"
    if not path.exists():
        return []
    row = next(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip() and json.loads(line)["id"] == sentence_id)
    hyp = row["hyp"].split()
    norm, ref = [_word(w) for w in hyp], [_word(w) for w in row["ref"].split()]
    joined = {a + b for a, b in itertools.pairwise(ref)}  # "kasetçalar" for "kaset çalar"
    ok = [not n or any(c.isdigit() for c in n) or n in joined
          or (n.startswith(UNIT_WORDS) and i > 0 and any(c.isdigit() for c in norm[i - 1]))
          for i, n in enumerate(norm)]
    for block in difflib.SequenceMatcher(a=norm, b=ref, autojunk=False).get_matching_blocks():
        for i in range(block.a, block.a + block.size):
            ok[i] = True
    return [{"w": w, "ok": o} for w, o in zip(hyp, ok, strict=True)]


def heard_all(sentence_id: str) -> dict[str, list[dict]]:
    """A word every system's transcript contains is a difference in how the reference is written (%3,5 for
    "yüzde 3,5", 75 kuruş for ",75 TL"), not a mishearing: never strike it."""
    words = {k: heard(k, sentence_id) for k in DEMO_SYSTEMS}
    common = set.intersection(*({_word(w["w"]) for w in ws} for ws in words.values())) if all(words.values()) else set()
    return {k: [{**w, "ok": w["ok"] or _word(w["w"]) in common} for w in ws] for k, ws in words.items()}


def metrics() -> None:
    utmos = json.loads((UTMOS / "utmos.json").read_text())
    hard_utmos = json.loads((HARD / "utmos.json").read_text()) if (HARD / "utmos.json").exists() else {}
    out = {}
    for system, meta in SYSTEMS.items():
        freya = FREYA_DIR.get(system, system)
        rate, device = rtf(system)
        extra = {"rtf_torch": rtf("ours_torch")[0]} if system == "ours" else {}
        row = {**meta, **extra, "device": device, "rtf": rate, "freya_dir": f"runs/utmos/{freya}",
               "utmos": utmos[freya]["utmos"], "utmos_ci95": utmos[freya]["ci95"], "utmos_n": utmos[freya]["n"]}
        if system in hard_utmos:
            row["utmos_hard16"] = hard_utmos[system]["utmos"]
        asr_freya = COMPARE / "asr" / "freya" / f"{freya}.jsonl"
        if asr_freya.exists():
            row |= error_rates_for(asr_freya)
        asr_hard = COMPARE / "asr" / "hard" / f"{system}.jsonl"
        if asr_hard.exists():
            h = error_rates_for(asr_hard)
            row |= {"wer_hard16": h["wer_all"], "cer_hard16": h["cer_all"]}
        out[system] = row
    cpu = next((json.loads((BENCH / k / "timing.json").read_text()).get("cpu") for k in SYSTEMS
                if (BENCH / k / "timing.json").exists()), "unknown")
    pd = paired(utmos["v02"]["per_file"], utmos["v02_dec110k"]["per_file"])
    pd["what"] = "UTMOS per Freya sentence, v02_dec110k (decoder 110k) minus v02 (decoder 70k), seed 0"
    notes = [
        f"Speed = seconds of audio per second of compute over the 16 hard sentences, every system on the same 8 cores "
        f"of one server CPU ({cpu}), model load and one warm-up excluded. On an Apple M4 our model runs 67x real time "
        "with MLX on the GPU (EMA Lightning: 18x on its CPU).",
        "Freya-TR-Eval: 495 sentences, seed 0 only, for every system (the 3-seed GPU numbers are not comparable). "
        "Whisper-large-v3, beam 5, 16 kHz input, the project's normalize(). clean = not in our training transcripts.",
        "UTMOS22 strong, mean over the 495 Freya renders; utmos_ci95 is the half-width of a normal 95% CI.",
        "Raw text in for every system. MMS has no text normalizer: its vocabulary lacks 5/7/8/9 and most punctuation, "
        "so digits are dropped or misread. MMS input gets only a Turkish I->ı / İ->i fix before its tokenizer, "
        "because the tokenizer's str.lower() maps I to i.",
        "Piper (VITS via onnxruntime) samples noise inside the ONNX graph and cannot be seeded; MMS uses "
        "torch.manual_seed(0) per sentence; ours and EMA Lightning use seed 0.",
        "Piper voices fahrettin and fettah were removed from rhasspy/piper-voices at their contributors' request and "
        "are not used.",
    ]
    (DEMO / "metrics.json").write_text(json.dumps({"systems": out, "paired_decoder": pd, "notes": notes}, indent=1,
                                                  ensure_ascii=False), encoding="utf-8")
    samples = [{**s, "audio": {sys_: f"audio/{sys_}/{s['id']}.mp3" for sys_ in DEMO_SYSTEMS},
                "heard": heard_all(s["id"])} for s in load_hard()]
    (DEMO / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'system':16s} {'UTMOS':>12s} {'WER':>6s} {'CER':>6s} {'WERc':>6s} {'CERc':>6s} {'RTF':>7s}  device")
    for k, r in out.items():
        def pct(x):
            return f"{100 * x:5.2f}%" if x is not None else "   -  "
        print(f"{k:16s} {r['utmos']:.3f}±{r['utmos_ci95']:.3f} {pct(r.get('wer_all'))} {pct(r.get('cer_all'))} "
              f"{pct(r.get('wer_clean'))} {pct(r.get('cer_clean'))} {r['rtf'] or 0:6.1f}x  {r['device']}")
    print("paired decoder:", pd)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "asr":
        asr(args[0], args[1:])
    elif cmd == "bench":
        bench(args[0])
    else:
        {"export": export, "render": render, "mp3": mp3, "metrics": metrics}[cmd]()
