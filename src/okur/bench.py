"""Measure every speed option on the current machine, on real prepared data: run first on a new GPU.

Acoustic training: eager / latents on device + padded shapes / torch.compile / CUDA graphs ("reduce-overhead").
Decoder GAN (adversarial phase): fp32 / bf16 / bf16 + compile.
Data prep: recognizer fp32 vs bf16 (speed, alignment drift), codec fp32 vs TF32 (speed, latent drift).
Throughput excludes warm-up (compilation, autotuning). Prints a table and writes bench.json in `out`.
"""

import io
import json
import logging
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

log = logging.getLogger(__name__)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _timed(step: Callable[[], float], warmup: int, steps: int, device: torch.device) -> tuple[float, float]:
    """Run `step` (returns work units, e.g. audio seconds) warmup + steps times; (units per second, peak GB)."""
    for _ in range(warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start, units = time.perf_counter(), 0.0
    for _ in range(steps):
        units += step()
    _sync(device)
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan")
    return units / (time.perf_counter() - start), peak


def acoustic(data: list[Path], max_frames: int, warmup: int, steps: int) -> list[dict[str, Any]]:
    from okur.train.config import TrainConfig
    from okur.train.data import Corpus
    from okur.train.trainer import Trainer

    corpus = Corpus(data)
    variants = [("eager", {}),
                ("device latents + padding", {"latents_on_device": True, "pad_frames": 64, "pad_letters": 32}),
                ("+ compile", {"latents_on_device": True, "pad_frames": 64, "pad_letters": 32, "compile": True}),
                ("+ CUDA graphs", {"latents_on_device": True, "pad_frames": 64, "pad_letters": 32, "compile": True,
                                   "compile_mode": "reduce-overhead"})]
    rows = []
    for name, extra in variants:
        with tempfile.TemporaryDirectory() as run_dir:
            cfg = TrainConfig.model_validate({"run_dir": run_dir, "data": data, "max_frames": max_frames,
                                              "max_letters": max_frames, "precision": "bf16", "sample_every": 0,
                                              "workers": 8, **extra})
            try:
                t = Trainer(cfg, corpus=corpus)
                t._reset_accumulators()

                def epochs(t: Trainer = t) -> Iterator[dict[str, torch.Tensor]]:
                    while True:  # small corpora run out of batches: start the next epoch
                        yield from t._batches()
                        t.epoch, t.batch_in_epoch = t.epoch + 1, 0

                batches = epochs()

                def step(t: Trainer = t, batches: Any = batches) -> float:
                    batch = next(batches)
                    t.train_step(batch)
                    t.step += 1
                    return float(batch["fmask"].sum()) / 25

                rate, peak = _timed(step, warmup, steps, t.device)
                rows.append({"stage": "acoustic", "variant": name, "audio_s_per_s": round(rate), "peak_gb": peak})
            except Exception as e:  # a variant that fails (e.g. unsupported compile mode) is reported, not fatal
                rows.append({"stage": "acoustic", "variant": name, "error": repr(e)[:200]})
            log.info("%s", rows[-1])
    return rows


def decoder(data: list[Path], batch: int, warmup: int, steps: int) -> list[dict[str, Any]]:
    from okur.train.decoder_trainer import DecoderTrainConfig, DecoderTrainer, Segments

    segments = Segments(data, 32, 0, held_out=4)
    rows = []
    for name, extra in [("fp32", {}), ("bf16", {"precision": "bf16"}),
                        ("bf16 + compile", {"precision": "bf16", "compile": True})]:
        with tempfile.TemporaryDirectory() as run_dir:
            cfg = DecoderTrainConfig.model_validate({"run_dir": run_dir, "data": data, "batch": batch,
                                                     "mel_only_steps": 0, "workers": 8, **extra})
            try:
                t = DecoderTrainer(cfg, data=segments)
                t._acc = torch.zeros(5, device=t.device)
                xs = [segments[(i, 0)] for i in range(batch)]
                z, audio = torch.stack([x[0] for x in xs]), torch.stack([x[1] for x in xs])

                def step(t: DecoderTrainer = t, z: torch.Tensor = z, audio: torch.Tensor = audio) -> float:
                    t.train_step(z, audio)
                    t.step += 1
                    return 1.0

                rate, peak = _timed(step, warmup, steps, t.device)
                rows.append({"stage": "decoder", "variant": name, "steps_per_s": round(rate, 2), "peak_gb": peak,
                             "hours_per_100k_steps": round(100_000 / rate / 3600, 1)})
            except Exception as e:
                rows.append({"stage": "decoder", "variant": name, "error": repr(e)[:200]})
            log.info("%s", rows[-1])
    return rows


def prep(data: list[Path], device: str, clips: int) -> list[dict[str, Any]]:
    from okur.data.align import Recognizer, align
    from okur.data.audio import resample
    from okur.data.codec import HOP, RATE, Codec
    from okur.data.shards import read_shards

    examples = read_shards(sorted(p for d in data for p in d.glob("shard-*.parquet"))[:1], audio=True)[:clips]
    audio48 = [sf.read(io.BytesIO(e.audio), dtype="float32")[0] for e in examples]
    audio48 = [a[: len(a) // HOP * HOP] for a in audio48]
    audio16 = [resample(a, RATE, 16000) for a in audio48]
    seconds = sum(len(a) for a in audio48) / RATE
    dev = torch.device(device)
    rows: list[dict[str, Any]] = []
    starts: dict[bool, list[np.ndarray]] = {}
    for bf16 in (False, True):
        rec = Recognizer(device, bf16=bf16)
        rec.emissions(audio16[:2])
        _sync(dev)
        t0 = time.perf_counter()
        ems = [e for i in range(0, len(audio16), 16) for e in rec.emissions(audio16[i:i + 16])]
        _sync(dev)
        el = time.perf_counter() - t0
        starts[bf16] = []
        for e, ex in zip(ems, examples, strict=True):
            ids, where = rec.tokens(ex.letters)
            a = align(e, ex.letters, ids, where, rec.blank)
            starts[bf16].append(a.starts if a else np.zeros(len(ex.letters)))
        rows.append({"stage": "prep", "variant": f"recognizer {'bf16' if bf16 else 'fp32'}",
                     "x_realtime": round(seconds / el)})
    drift = np.mean([np.abs(a - b).mean() for a, b in zip(starts[False], starts[True], strict=True)]) * 1000
    rows[-1]["alignment_drift_ms"] = round(float(drift), 1)
    ref = None
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        codec = Codec(device)
        codec.encode(audio48[:1])
        _sync(dev)
        t0 = time.perf_counter()
        z = [x for i in range(0, len(audio48), 8) for x in codec.encode(audio48[i:i + 8])]
        _sync(dev)
        el = time.perf_counter() - t0
        row: dict[str, Any] = {"stage": "prep", "variant": f"codec {'tf32' if tf32 else 'fp32'}",
                               "x_realtime": round(seconds / el)}
        if ref is None:
            ref = z
        else:
            row["latent_max_diff"] = round(max(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())
                                               for a, b in zip(ref, z, strict=True)), 4)
        rows.append(row)
    return rows


def run(data: list[Path], out: Path, *, device: str, max_frames: int, decoder_batch: int, warmup: int, steps: int,
        clips: int) -> None:
    rows = prep(data, device, clips) + acoustic(data, max_frames, warmup, steps) + \
        decoder(data, decoder_batch, warmup, steps)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(rows, indent=1))
    keys = sorted({k for r in rows for k in r} - {"stage", "variant"})
    print(f"\n{'stage':10s} {'variant':28s} " + " ".join(f"{k:>22s}" for k in keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            cells.append(f"{v:>22.2f}" if isinstance(v, float) else f"{v!s:>22s}"[:22])
        print(f"{r['stage']:10s} {r['variant']:28s} " + " ".join(cells))
