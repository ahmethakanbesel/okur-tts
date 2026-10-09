"""Raw corpus → training shards: decode, normalize text, align letters, crop silence, encode latents.

Fault tolerant: work is split into fixed shards and every finished shard is written atomically, so a crashed or
pre-empted run resumes by skipping shards that exist. Bad clips are counted per reason, never fatal.

Parallel: a process pool decodes and aligns on every CPU core while the GPU runs the recognizer and the codec; the next
shard is decoded while the current one is on the GPU. With several GPUs, run one process per GPU with --rank/--world;
shard i belongs to rank i % world.
"""

import io
import logging
import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from jaxtyping import Float32

from okur.data.align import Alignment, Recognizer, align
from okur.data.audio import decode, quality_bucket, resample
from okur.data.codec import HOP, LATENT_RATE, RATE, Codec
from okur.data.shards import Example, shard_path, write_shard
from okur.data.sources import SOURCES, RawItem, RawRef

log = logging.getLogger(__name__)

LEAD, TAIL = 0.15, 0.20  # seconds of silence kept before the first and after the last letter
MIN_FRAMES, MAX_FRAMES = 25, 500  # 1 to 20 seconds
MIN_SCORE = -1.0  # mean log-probability of aligned letters; below this the transcript does not match the audio
MAX_LETTER_FRAMES = 40  # a single non-space letter lasting over 1.6 s means a bad alignment


@dataclass(frozen=True)
class Decoded:
    item: RawItem
    letters: str
    quality: int
    audio16: Float32[np.ndarray, "..."]
    audio48: Float32[np.ndarray, "..."]


_frontend = None


def _init_worker() -> None:
    import torch

    from okur.frontend import Frontend

    global _frontend  # noqa: PLW0603 -- one frontend per pool worker
    _frontend = Frontend()
    torch.set_num_threads(1)


def _decode(item: RawItem) -> Decoded | str:
    try:
        audio, rate = decode(item.audio)
    except Exception:
        return "undecodable"
    assert _frontend is not None
    letters = _frontend(item.text)
    if not letters:
        return "empty_text"
    return Decoded(item, letters, quality_bucket(rate), resample(audio, rate, 16000), resample(audio, rate, RATE))


def _align(args: tuple[Float32[np.ndarray, "..."], str, list[int], list[int], int]) -> Alignment | None:
    return align(*args)


def _crop(d: Decoded, a: Alignment) -> tuple[Float32[np.ndarray, "..."], Float32[np.ndarray, "..."]] | str:
    """Cut to the speech plus a little silence, a whole number of latent frames; letter durations in frames."""
    if a.score < MIN_SCORE:
        return "low_score"
    start = max(0.0, a.starts[0] - LEAD)
    stop = min(len(d.audio48) / RATE, a.end + TAIL)
    n_frames = int((stop - start) * LATENT_RATE)
    if not MIN_FRAMES <= n_frames <= MAX_FRAMES:
        return "length"
    s = round(start * RATE)
    audio = d.audio48[s:s + n_frames * HOP]
    if len(audio) < n_frames * HOP:
        return "length"
    bounds = np.clip((a.starts - start) * LATENT_RATE, 0, n_frames)
    bounds[0] = 0.0
    bounds = np.maximum.accumulate(np.append(bounds, n_frames))
    dur = np.diff(bounds).astype(np.float32)
    spoken = np.array([c != " " for c in d.letters])
    if (dur[spoken][:-1] > MAX_LETTER_FRAMES).any():
        return "long_letter"
    return audio, dur


def _by_duration(kept: list[tuple[Decoded, Float32[np.ndarray, "..."], Float32[np.ndarray, "..."], float]],
                 seconds: float) -> list[list[int]]:
    """Consecutive batches of clips (already sorted by length) holding at most `seconds` of padded audio: the
    codec's memory grows with batch x longest clip, so a fixed clip count overflows on long clips."""
    batches: list[list[int]] = []
    batch: list[int] = []
    for i, k in enumerate(kept):
        longest = len(k[1]) / RATE  # sorted ascending: the newest clip is the longest
        if batch and (len(batch) + 1) * longest > seconds:
            batches.append(batch)
            batch = []
        batch.append(i)
    if batch:
        batches.append(batch)
    return batches


def _flac(audio: Float32[np.ndarray, "..."]) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), RATE, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def prepare(source: str, raw_root: Path, out_dir: Path, *, shard_size: int = 128, device: str = "mps",
            workers: int = 8, rank: int = 0, world: int = 1, limit: int | None = None, align_batch: int = 8,
            encode_seconds: float = 15.0, tf32: bool = True, recognizer_bf16: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = SOURCES[source](raw_root)
    refs = src.refs()[:limit]
    shards = [refs[i:i + shard_size] for i in range(0, len(refs), shard_size)]
    mine = [(i, s) for i, s in enumerate(shards) if i % world == rank and not shard_path(out_dir, i).exists()]
    log.info("%s: %d clips in %d shards; %d left for rank %d", source, len(refs), len(shards), len(mine), rank)
    if not mine:
        return
    if device.startswith("cuda"):  # TF32 keeps fp32 range with a 10-bit mantissa; fp16 changed latents too much
        import torch

        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    recognizer, codec = Recognizer(device, bf16=recognizer_bf16), Codec(device)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as pool, \
            ThreadPoolExecutor(1) as prefetch:

        def decode_shard(shard_refs: list[RawRef]) -> list[Decoded | str]:
            return list(pool.map(_decode, src.load(shard_refs), chunksize=4))

        upcoming = prefetch.submit(decode_shard, mine[0][1])
        started, done_seconds = time.perf_counter(), 0.0
        for k, (index, shard_refs) in enumerate(mine):
            decoded = upcoming.result()
            if k + 1 < len(mine):
                upcoming = prefetch.submit(decode_shard, mine[k + 1][1])
            stats: Counter[str] = Counter(r for r in decoded if isinstance(r, str))
            good = sorted((d for d in decoded if isinstance(d, Decoded)), key=lambda d: len(d.audio16))

            emissions = [e for i in range(0, len(good), align_batch)
                         for e in recognizer.emissions([d.audio16 for d in good[i:i + align_batch]])]
            jobs = [(e, d.letters, *recognizer.tokens(d.letters), recognizer.blank)
                    for e, d in zip(emissions, good, strict=True)]
            alignments = list(pool.map(_align, jobs, chunksize=2))

            kept: list[tuple[Decoded, Float32[np.ndarray, ...], Float32[np.ndarray, ...], float]] = []
            for d, a in zip(good, alignments, strict=True):
                cropped = "no_alignment" if a is None else _crop(d, a)
                if isinstance(cropped, str):
                    stats[cropped] += 1
                else:
                    assert a is not None
                    kept.append((d, *cropped, a.score))
            latents = [z for batch in _by_duration(kept, encode_seconds)
                       for z in codec.encode([kept[i][1] for i in batch])]
            examples = [Example(d.item.id, d.item.source, d.item.speaker, d.quality, d.item.text, d.letters, dur, z,
                                score, _flac(audio)) for (d, audio, dur, score), z in zip(kept, latents, strict=True)]
            stats["kept"] = len(examples)
            write_shard(out_dir, index, examples, dict(stats))

            done_seconds += sum(len(z) for z in latents) / LATENT_RATE
            elapsed = time.perf_counter() - started
            log.info("shard %d (%d/%d): kept %d/%d %s | %.0fx realtime, ETA %.0f min", index, k + 1, len(mine),
                     len(examples), len(shard_refs), dict(stats), done_seconds / elapsed,
                     elapsed / (k + 1) * (len(mine) - k - 1) / 60)
