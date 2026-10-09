"""Training batches: length-bucketed to a frame budget, deterministic per epoch, resumable, split across ranks.

All prepared latents fit in memory (about 4 GB per 300 h at float16) and are kept as one flat buffer, so there is no
I/O during training. Each clip's word–letter timeline is computed once at load. A batch is then only padding, done by
DataLoader workers in parallel with the GPU.

With `latents_on_device`, the flat latent buffer is copied to the GPU once and batches carry only frame indices
(`lat_idx`); the trainer gathers latents on the GPU. This removes the largest host-to-device copy of every step.

Padding lengths up to multiples (`pad_frames`, `pad_letters`) bounds the number of distinct shapes, so torch.compile
and CUDA graphs reuse compiled code instead of recompiling for every length. Padding is masked: it changes no result.
"""

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from okur.frontend import encode
from okur.model.timeline import frame_timeline, word_frames, words

POOL = 64  # batches drawn from a pool this many batches wide, so lengths group without fixing the order


@dataclass(frozen=True)
class Item:
    ids: np.ndarray  # int64 (letters,)
    cw: np.ndarray  # int64 (letters,)
    wstart: np.ndarray  # int64 (letters,)
    dur: np.ndarray  # float32 (letters,)
    fw: np.ndarray  # int64 (frames,)
    fp: np.ndarray  # float32 (frames,)
    offset: int  # first frame in the corpus' flat latent buffer
    speaker: int
    quality: int


class Corpus(Dataset[Item]):
    def __init__(self, dirs: Sequence[Path], speakers: list[str] | None = None,
                 only: Sequence[str] | None = None) -> None:
        from okur.data.shards import read_shards

        examples = read_shards(sorted(p for d in dirs for p in d.glob("shard-*.parquet")))
        if not examples:
            raise FileNotFoundError(f"no shards in {list(dirs)}")
        self.speakers = speakers or sorted({e.speaker for e in examples})
        index = {s: i for i, s in enumerate(self.speakers)}
        examples = [e for e in examples if e.speaker in index and (only is None or e.speaker in only)]
        if not examples:
            raise ValueError(f"no examples for speakers {only}")
        self.latents = np.concatenate([e.latents for e in examples])  # float16 (total frames, 64)
        self.items: list[Item] = []
        offset = 0
        for e in examples:
            w = words(e.letters)
            dur = torch.from_numpy(e.dur)
            fw, fp = frame_timeline(word_frames(dur, w.cw, w.n_words))
            assert len(fw) == len(e.latents), (e.id, len(fw), len(e.latents))
            self.items.append(Item(np.asarray(encode(e.letters), dtype=np.int64), w.cw.numpy(), w.wstart.numpy(),
                                   e.dur, fw.numpy(), fp.numpy(), offset, index[e.speaker], e.quality))
            offset += len(e.latents)
        self.frames = np.array([len(i.fw) for i in self.items])
        self.letters = np.array([len(i.ids) for i in self.items])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Item:
        return self.items[i]

    def latent_stats(self) -> tuple[Tensor, Tensor]:
        sample = self.latents[:: max(1, len(self.latents) // 500_000)].astype(np.float32)
        return torch.from_numpy(sample.mean(0)), torch.from_numpy(sample.std(0).clip(1e-3))


class BucketSampler(Sampler[list[int]]):
    """Batches of similar length up to `max_frames` latent frames. The batch list depends only on (seed, epoch), so a
    resumed run continues at batch `start` of the same epoch. Each rank takes every world-th batch."""

    def __init__(self, frames: np.ndarray, letters: np.ndarray, max_frames: int, max_letters: int, seed: int,
                 rank: int = 0, world: int = 1) -> None:
        self.frames, self.letters = frames, letters
        self.max_frames, self.max_letters = max_frames, max_letters
        self.seed, self.rank, self.world = seed, rank, world
        self.epoch, self.start = 0, 0

    def batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed * 1_000_003 + epoch)
        order = list(range(len(self.frames)))
        rng.shuffle(order)
        out: list[list[int]] = []
        pool = POOL * max(1, int(self.max_frames // max(1.0, self.frames.mean())))
        for p in range(0, len(order), pool):
            chunk = sorted(order[p:p + pool], key=lambda i: self.frames[i])
            batch: list[int] = []
            longest = widest = 0
            for i in chunk:  # sorted by frames, so the newest item is the longest
                longest, widest = int(self.frames[i]), max(widest, int(self.letters[i]))
                n = len(batch) + 1
                if batch and (n * longest > self.max_frames or n * widest > self.max_letters):
                    out.append(batch)
                    batch, widest = [], int(self.letters[i])
                batch.append(i)
            if batch:
                out.append(batch)
        rng.shuffle(out)
        usable = len(out) // self.world * self.world  # every rank gets the same number of steps
        return out[self.rank:usable:self.world]

    def set_position(self, epoch: int, start: int) -> None:
        self.epoch, self.start = epoch, start

    def __len__(self) -> int:
        return len(self.batches(self.epoch)) - self.start

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches(self.epoch)[self.start:]


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


class Collate:
    """Pads a list of items into a batch. Picklable, so DataLoader workers can run it."""

    def __init__(self, latents: np.ndarray, *, pad_frames: int = 1, pad_letters: int = 1,
                 latents_on_device: bool = False) -> None:
        self.latents, self.pad_frames, self.pad_letters = latents, pad_frames, pad_letters
        self.on_device = latents_on_device

    def __call__(self, items: list[Item]) -> dict[str, Tensor]:
        n_letters = _round_up(max(len(i.ids) for i in items), self.pad_letters)
        n_frames = _round_up(max(len(i.fw) for i in items), self.pad_frames)
        b = len(items)
        ids = np.zeros((b, n_letters), np.int64)
        cw = np.full((b, n_letters), -1, np.int64)
        wstart = np.zeros((b, n_letters), np.int64)
        dur = np.zeros((b, n_letters), np.float32)
        fw = np.full((b, n_frames), -1, np.int64)
        fp = np.zeros((b, n_frames), np.float32)
        lat_idx = np.zeros((b, n_frames), np.int64)
        for j, it in enumerate(items):
            n, t = len(it.ids), len(it.fw)
            ids[j, :n], cw[j, :n], wstart[j, :n], dur[j, :n] = it.ids, it.cw, it.wstart, it.dur
            fw[j, :t], fp[j, :t] = it.fw, it.fp
            lat_idx[j, :t] = np.arange(it.offset, it.offset + t)
        out = {"ids": torch.from_numpy(ids), "cw": torch.from_numpy(cw), "wstart": torch.from_numpy(wstart),
               "dur": torch.from_numpy(dur), "fw": torch.from_numpy(fw), "fp": torch.from_numpy(fp),
               "speaker": torch.tensor([i.speaker for i in items]), "quality": torch.tensor([i.quality for i in items])}
        out["mask"] = out["ids"] != 0
        out["fmask"] = out["fw"] >= 0
        if self.on_device:
            out["lat_idx"] = torch.from_numpy(lat_idx)
        else:
            lat = self.latents[lat_idx].astype(np.float32)
            lat[fw < 0] = 0.0
            out["latents"] = torch.from_numpy(lat)
        return out
