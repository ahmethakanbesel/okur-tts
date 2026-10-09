"""Decoder training: a GAN that turns codec latents into 48 kHz audio, learning from the real recordings.

Same fault-tolerance contract as the acoustic trainer: automatic resume from the newest readable checkpoint,
background atomic checkpoints, clean stop on SIGTERM/SIGINT/SIGUSR1, random state derived from (seed, step, rank),
non-finite steps skipped. Crop positions come from (seed, epoch, clip), so a resumed run sees the same audio.
Parallel: torchrun DDP across GPUs, DataLoader workers decode FLAC while the GPU trains.
Speed: bf16 autocast for generator and discriminators (mel loss in fp32), fused AdamW skipping non-finite steps on the
device, metrics read once per log interval, cudnn autotuning, optional torch.compile (crop shapes are fixed).
Held-out clips are decoded to samples/ every `sample_every` steps, to pick the stopping point by ear.
"""

import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from okur.data.codec import HOP, RATE
from okur.data.shards import atomic_write_bytes, read_shards
from okur.model.config import DecoderConfig
from okur.model.decoder import Decoder
from okur.model.discriminators import Discriminators, gan_losses, mel_loss
from okur.train.checkpoint import Checkpointer
from okur.train.trainer import _Stop

log = logging.getLogger(__name__)


class DecoderTrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_dir: Path
    data: list[Path]
    model: DecoderConfig = DecoderConfig()
    steps: Annotated[int, Field(gt=0)] = 300_000
    lr: float = 2e-4
    lr_decay: float = 0.999_99  # per step
    batch: int = 16
    segment_frames: int = 32  # 1.28 s crops
    mel_weight: float = 45.0
    mel_only_steps: int = 5_000  # reconstruction-only warm-up before the discriminators join
    workers: int = 4
    seed: int = 0
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    precision: Literal["fp32", "bf16"] = "fp32"
    compile: bool = False
    sample_every: int = 0
    val_clips: int = 4  # held out from training, decoded at every sample step
    ckpt_every_steps: int = 2_000
    ckpt_every_minutes: float = 15.0
    keep_checkpoints: int = 3
    log_every: int = 50

    @classmethod
    def load(cls, path: Path) -> "DecoderTrainConfig":
        return cls.model_validate(yaml.safe_load(path.read_text()))


class Segments(Dataset[tuple[Tensor, Tensor]]):
    """(latent crop, matching audio crop). Index = (clip, epoch) so the crop position is reproducible."""

    def __init__(self, dirs: list[Path], segment: int, seed: int, held_out: int = 0) -> None:
        examples = [e for e in read_shards(sorted(p for d in dirs for p in d.glob("shard-*.parquet")), audio=True)
                    if e.audio and len(e.latents) > segment]
        if len(examples) <= held_out:
            raise FileNotFoundError(f"no shards with audio in {dirs}")
        self.held_out = examples[len(examples) - held_out:]
        examples = examples[:len(examples) - held_out]
        self.latents = [e.latents for e in examples]
        self.audio = [e.audio for e in examples]
        self.segment, self.seed = segment, seed

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, key: tuple[int, int]) -> tuple[Tensor, Tensor]:
        clip, epoch = key
        z = self.latents[clip]
        start = random.Random(self.seed * 1_000_003 + epoch * 7919 + clip).randrange(len(z) - self.segment)
        audio, _ = sf.read(io.BytesIO(self.audio[clip]), dtype="float32", start=start * HOP,
                           stop=(start + self.segment) * HOP)
        return torch.from_numpy(z[start:start + self.segment].astype(np.float32)).T, torch.from_numpy(audio)


class EpochSampler(Sampler[list[tuple[int, int]]]):
    def __init__(self, n: int, batch: int, seed: int, rank: int, world: int) -> None:
        self.n, self.batch, self.seed, self.rank, self.world = n, batch, seed, rank, world
        self.epoch, self.start = 0, 0

    def batches(self, epoch: int) -> list[list[tuple[int, int]]]:
        order = list(range(self.n))
        random.Random(self.seed + epoch).shuffle(order)
        full = len(order) // (self.batch * self.world) * self.batch * self.world
        out = [[(i, epoch) for i in order[b:b + self.batch]] for b in range(0, full, self.batch)]
        return out[self.rank::self.world]

    def __len__(self) -> int:
        return len(self.batches(self.epoch)) - self.start

    def __iter__(self):  # type: ignore[override]
        yield from self.batches(self.epoch)[self.start:]


def _stack(items: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    return torch.stack([z for z, _ in items]), torch.stack([a for _, a in items])


class DecoderTrainer:
    def __init__(self, cfg: DecoderTrainConfig, data: "Segments | None" = None) -> None:
        self.cfg = cfg
        self.rank, self.world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world > 1:
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if cfg.device == "auto":
            self.device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device, local) if cfg.device == "cuda" else torch.device(cfg.device)
        self.main = self.rank == 0
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = Checkpointer(cfg.run_dir, cfg.keep_checkpoints)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # fixed crop shapes: autotune convolution algorithms once
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed)
        self.gen = Decoder(cfg.model).to(self.device)
        self.disc = Discriminators().to(self.device)
        self.opt_g = torch.optim.AdamW(self.gen.parameters(), cfg.lr, betas=(0.8, 0.99), fused=True)
        self.opt_d = torch.optim.AdamW(self.disc.parameters(), cfg.lr, betas=(0.8, 0.99), fused=True)
        self.step = self.epoch = self.batch_in_epoch = 0
        state = self.ckpt.load_latest()
        if state is not None:
            self.gen.load_state_dict(state["model"])
            self.disc.load_state_dict(state["disc"])
            self.opt_g.load_state_dict(state["opt_g"])
            self.opt_d.load_state_dict(state["opt_d"])
            self.step, self.epoch, self.batch_in_epoch = state["step"], state["epoch"], state["batch_in_epoch"]
        self.data = data or Segments(cfg.data, cfg.segment_frames, cfg.seed, held_out=cfg.val_clips)
        self.sampler = EpochSampler(len(self.data), cfg.batch, cfg.seed, self.rank, self.world)
        self.g_net: Any = self.gen
        self.d_net: Any = self.disc
        if self.world > 1:
            ids = [local] if self.device.type == "cuda" else None
            self.g_net, self.d_net = DistributedDataParallel(self.gen, ids), DistributedDataParallel(self.disc, ids)
        if cfg.compile:
            self.g_net, self.d_net = torch.compile(self.g_net), torch.compile(self.d_net)
        if self.main:
            log.info("%d clips | decoder %.2fM params, discriminators %.1fM | %s x%d | step %d", len(self.data),
                     sum(p.numel() for p in self.gen.parameters()) / 1e6,
                     sum(p.numel() for p in self.disc.parameters()) / 1e6, self.device, self.world, self.step)

    def _state(self) -> dict[str, Any]:
        return {"step": self.step, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                "model": self.gen.state_dict(), "disc": self.disc.state_dict(), "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(), "model_config": self.cfg.model.model_dump(),
                "train_config": self.cfg.model_dump(mode="json")}

    @staticmethod
    def _step(opt: torch.optim.AdamW, params: Any, loss: Tensor) -> Tensor:
        """Backward, clip, and a fused AdamW step that the device skips when the gradients are not finite."""
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1e3)
        bad = (~torch.isfinite(norm)).float()
        opt.found_inf = bad  # pyright: ignore[reportAttributeAccessIssue]
        opt.grad_scale = None  # pyright: ignore[reportAttributeAccessIssue]
        opt.step()
        return bad

    def train_step(self, z: Tensor, audio: Tensor) -> None:
        cfg = self.cfg
        torch.manual_seed((cfg.seed * 1_000_003 + self.step) * self.world + self.rank)
        lr = cfg.lr * cfg.lr_decay ** self.step
        for o in (self.opt_g, self.opt_d):
            for g in o.param_groups:
                g["lr"] = lr
        z, audio = z.to(self.device, non_blocking=True), audio.to(self.device, non_blocking=True)
        bf16 = cfg.precision == "bf16" and self.device.type == "cuda"
        adversarial = self.step >= cfg.mel_only_steps
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
            fake = self.g_net(z)
        fake = fake.float()
        d_loss = torch.zeros((), device=self.device)
        bad_d = torch.zeros((), device=self.device)
        if adversarial:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
                d_loss, _ = gan_losses(self.d_net(audio), self.d_net(fake.detach()))
            bad_d = self._step(self.opt_d, self.disc.parameters(), d_loss.float())
        mel = mel_loss(fake, audio)  # fp32: STFTs need it
        g_loss = cfg.mel_weight * mel
        if adversarial:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
                g_loss = g_loss + gan_losses(self.d_net(audio), self.d_net(fake))[1].float()
        bad_g = self._step(self.opt_g, self.gen.parameters(), g_loss)
        with torch.no_grad():
            good = 1 - bad_g
            self._acc += torch.stack([torch.nan_to_num(mel.detach()) * good, torch.nan_to_num(g_loss.detach()) * good,
                                      torch.nan_to_num(d_loss.detach().float()) * good, good, bad_g + bad_d])
        self._lr = lr

    @torch.no_grad()
    def sample(self) -> None:
        """Decode the held-out clips in full; write wavs and log their mel loss against the real recordings."""
        out = self.cfg.run_dir / "samples"
        out.mkdir(exist_ok=True)
        self.gen.eval()
        losses = []
        for i, e in enumerate(self.data.held_out):
            z = torch.from_numpy(e.latents.astype(np.float32)).T[None].to(self.device)
            fake = self.gen(z)[0].float().cpu()
            real, _ = sf.read(io.BytesIO(e.audio), dtype="float32")
            n = min(len(real), len(fake))
            losses.append(mel_loss(fake[None, :n], torch.from_numpy(real[:n])[None]).item())
            sf.write(out / f"step-{self.step:08d}-{i}.wav", fake.numpy(), RATE)
        self.gen.train()
        log.info("step %d | held-out mel %.4f", self.step, sum(losses) / len(losses))

    def run(self) -> None:
        cfg = self.cfg
        _Stop.install()
        last_save, saved = time.monotonic(), self.step
        self._acc = torch.zeros(5, device=self.device)  # mel, g, d (good steps), good steps, skipped updates
        while self.step < cfg.steps and not _Stop.requested:
            self.sampler.epoch, self.sampler.start = self.epoch, self.batch_in_epoch
            loader = DataLoader(self.data, batch_sampler=self.sampler, collate_fn=_stack, num_workers=cfg.workers,
                                pin_memory=self.device.type == "cuda", prefetch_factor=4 if cfg.workers else None,
                                multiprocessing_context="fork" if cfg.workers else None,
                                worker_init_fn=_Stop.ignore_in_worker if cfg.workers else None)
            for z, audio in loader:
                self.train_step(z, audio)
                self.batch_in_epoch += 1
                self.step += 1
                if self.main and self.step % cfg.log_every == 0:
                    mel, g, d, good, skipped = self._acc.tolist()  # one host sync per log interval
                    self._acc.zero_()
                    avg = {k: (v / good if good else float("nan")) for k, v in (("mel", mel), ("g", g), ("d", d))}
                    record = json.dumps({"step": self.step, **avg, "skipped": int(skipped), "lr": self._lr,
                                         "time": time.time()})
                    with open(cfg.run_dir / "metrics.jsonl", "a") as f:
                        f.write(record + "\n")
                    atomic_write_bytes(cfg.run_dir / "heartbeat.json", record.encode())
                    log.info("step %d | mel %.4f g %.3f d %.3f | skipped %d", self.step, avg["mel"], avg["g"],
                             avg["d"], int(skipped))
                if self.main and cfg.sample_every and self.step % cfg.sample_every == 0:
                    self.sample()
                due = time.monotonic() - last_save > cfg.ckpt_every_minutes * 60
                if self.main and (self.step % cfg.ckpt_every_steps == 0 or due or _Stop.requested):
                    self.ckpt.save(self.step, self._state())
                    last_save, saved = time.monotonic(), self.step
                if _Stop.requested or self.step >= cfg.steps:
                    break
            else:
                self.epoch += 1
                self.batch_in_epoch = 0
        if self.main:
            if self.step != saved:
                self.ckpt.save(self.step, self._state())
            self.ckpt.wait()
            if self.step >= cfg.steps:
                (cfg.run_dir / "DONE").write_text(str(self.step))
        if self.world > 1:
            dist.destroy_process_group()


def main(config: Path) -> None:
    DecoderTrainer(DecoderTrainConfig.load(config)).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    main(Path(sys.argv[1]))

