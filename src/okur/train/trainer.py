"""Acoustic-model training loop, built to be killed at any moment and resumed exactly.

Fault tolerance:
- Resumes automatically from the newest readable checkpoint in the run directory (model, weight average, optimizer,
  data position, LR scale). Checkpoints every N steps and every M minutes, in the background.
- SIGTERM / SIGINT / SIGUSR1 (pre-emption notices) finish the current step, save, and exit cleanly.
- Random state is derived from (seed, step, rank) at every step, so a resumed run computes exactly what an
  uninterrupted run would have.
- Non-finite gradients skip the step on the GPU (fused AdamW's found_inf, no host sync); too many in a row roll back
  to the last checkpoint at half the learning rate. Checked at every log interval.
- A heartbeat file lets an external supervisor detect hangs; a DONE file marks completion.

Parallelism: one process per GPU under torchrun (DistributedDataParallel); DataLoader workers build batches in
parallel with the GPU; checkpoints are written by a background thread.

Speed: no host-device synchronization inside a step (metrics accumulate on the GPU and are read once per log
interval); optional latents kept on the GPU; optional torch.compile, including CUDA graphs ("reduce-overhead").
"""

import copy
import json
import logging
import math
import os
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from okur.data.shards import atomic_write_bytes
from okur.model.acoustic import Acoustic
from okur.model.config import AcousticConfig
from okur.train.checkpoint import Checkpointer
from okur.train.config import TrainConfig
from okur.train.data import BucketSampler, Collate, Corpus

log = logging.getLogger(__name__)


class _Stop:
    requested = False

    @classmethod
    def install(cls) -> None:
        def handler(signum: int, _frame: object) -> None:
            log.warning("signal %d: saving and stopping after this step", signum)
            cls.requested = True

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, handler)

    @staticmethod
    def ignore_in_worker(_worker_id: int) -> None:
        """DataLoader workers get the same SIGTERM as the trainer on pre-emption; leave the shutdown to the trainer."""
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, signal.SIG_IGN)


def _device(cfg: TrainConfig, local_rank: int) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank)
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(cfg.device, local_rank) if cfg.device == "cuda" else torch.device(cfg.device)


def _lr(cfg: TrainConfig, step: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def _optimizers(model: nn.Module, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    """AdamW everywhere, or Muon for the hidden weight matrices and AdamW for embeddings, norms, biases and I/O."""
    decay, no_decay, matrices = [], [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or "emb" in name or "speaker" in name or "quality" in name or "norm" in name:
            no_decay.append(p)
        elif cfg.optimizer == "muon" and p.ndim == 2 and not name.startswith(("in_proj", "out_proj")):
            matrices.append(p)
        else:
            decay.append(p)
    adamw = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.98), fused=True)
    if not matrices:
        return [adamw]
    return [adamw, torch.optim.Muon(matrices, lr=cfg.lr, weight_decay=cfg.weight_decay, adjust_lr_fn="match_rms_adamw")]


class Trainer:
    def __init__(self, cfg: TrainConfig, corpus: Corpus | None = None) -> None:
        self.cfg = cfg
        self.rank, self.world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world > 1:
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        self.device = _device(cfg, self.local_rank)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.main = self.rank == 0
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = Checkpointer(cfg.run_dir, cfg.keep_checkpoints)
        state = self.ckpt.load_latest()
        init = None
        if state is None and cfg.init_from is not None:  # fine-tuning: weights and speaker table from a checkpoint
            init = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        speakers = state["speakers"] if state else init["speakers"] if init else None
        self.corpus = corpus or Corpus(cfg.data, speakers=speakers, only=cfg.only_speakers)
        base = AcousticConfig.model_validate(init["model_config"]) if init else cfg.model  # fine-tune keeps its shape
        model_cfg = base.model_copy(update={"n_speakers": len(self.corpus.speakers)})
        torch.manual_seed(cfg.seed)  # identical initial weights on every rank and every fresh run
        self.model = Acoustic(model_cfg).to(self.device)
        if init is not None:
            self.model.load_state_dict(init["average"])  # keeps the base model's latent normalization
        elif state is None:
            mean, std = self.corpus.latent_stats()
            self.model.latent_mean.copy_(mean)
            self.model.latent_std.copy_(std)
        self.average = copy.deepcopy(self.model).requires_grad_(False)
        self.optimizers = _optimizers(self.model, cfg)
        self.step = self.epoch = self.batch_in_epoch = self.stale_evals = 0
        self.lr_scale, self.best_cer, self.stopped_early = 1.0, math.inf, False
        self.latents = torch.from_numpy(self.corpus.latents).to(self.device) if cfg.latents_on_device else None
        if state is not None:
            self._restore(state)
        self.sampler = BucketSampler(self.corpus.frames, self.corpus.letters, cfg.max_frames, cfg.max_letters,
                                     cfg.seed, self.rank, self.world)
        net: nn.Module = self.model
        if self.world > 1:
            net = DistributedDataParallel(net, device_ids=[self.local_rank] if self.device.type == "cuda" else None)
        if cfg.compile:
            torch._dynamo.config.cache_size_limit = 64  # one entry per padded shape bucket
        self.net = torch.compile(net, mode=cfg.compile_mode) if cfg.compile else net
        if self.main:
            n = sum(p.numel() for p in self.model.parameters())
            log.info("%d clips, %.1f h, %d speakers | %.2fM params | %s x%d | resumed at step %d",
                     len(self.corpus), self.corpus.frames.sum() / 25 / 3600, len(self.corpus.speakers), n / 1e6,
                     self.device, self.world, self.step)
            (cfg.run_dir / "config.json").write_text(cfg.model_dump_json(indent=1))

    # ---- state ------------------------------------------------------------------------------------------------------

    def _state(self) -> dict[str, Any]:
        return {"step": self.step, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                "lr_scale": self.lr_scale, "best_cer": self.best_cer, "stale_evals": self.stale_evals,
                "model": self.model.state_dict(), "average": self.average.state_dict(),
                "optimizers": [o.state_dict() for o in self.optimizers], "speakers": self.corpus.speakers,
                "model_config": self.model.cfg.model_dump(), "train_config": self.cfg.model_dump(mode="json")}

    def _restore(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.average.load_state_dict(state["average"])
        for o, s in zip(self.optimizers, state["optimizers"], strict=True):
            o.load_state_dict(s)
        self.step, self.epoch, self.batch_in_epoch = state["step"], state["epoch"], state["batch_in_epoch"]
        self.lr_scale = state["lr_scale"]
        self.best_cer, self.stale_evals = state.get("best_cer", math.inf), state.get("stale_evals", 0)

    def _save(self, *, wait: bool = False) -> None:
        if self.main and self.step != self._saved_step:
            self.ckpt.save(self.step, self._state(), wait=wait)
        elif self.main and wait:
            self.ckpt.wait()
        self._saved_step, self._last_save = self.step, time.monotonic()

    def _rollback(self) -> None:
        self.ckpt.wait()
        state = self.ckpt.load_latest()
        if state is None:
            raise RuntimeError("training diverged before the first checkpoint")
        self._restore(state)
        self._saved_step = self.step
        self.lr_scale *= 0.5
        self._reset_accumulators()
        log.warning("rolled back to step %d with lr scale %.3g", self.step, self.lr_scale)

    # ---- loop -------------------------------------------------------------------------------------------------------

    def _seed(self) -> None:
        torch.manual_seed((self.cfg.seed * 1_000_003 + self.step) * self.world + self.rank)

    def _loader(self) -> DataLoader[Any]:
        self.sampler.set_position(self.epoch, self.batch_in_epoch)
        workers = self.cfg.workers
        collate = Collate(self.corpus.latents, pad_frames=self.cfg.pad_frames, pad_letters=self.cfg.pad_letters,
                          latents_on_device=self.cfg.latents_on_device)
        return DataLoader(self.corpus, batch_sampler=self.sampler, collate_fn=collate, num_workers=workers,
                          pin_memory=self.device.type == "cuda", prefetch_factor=4 if workers else None,
                          persistent_workers=False, multiprocessing_context="fork" if workers else None,
                          worker_init_fn=_Stop.ignore_in_worker if workers else None)

    def _batches(self) -> Iterator[dict[str, torch.Tensor]]:
        """Batches of the current epoch. If the data pipeline fails, save first: between steps the state is whole."""
        it = iter(self._loader())
        while True:
            try:
                batch = next(it)
            except StopIteration:
                return
            except Exception:
                log.exception("data loading failed at step %d; saving before exiting", self.step)
                self._save(wait=True)
                raise
            yield batch

    def _reset_accumulators(self) -> None:
        # [loss, flow, dur, grad_norm, good steps, skipped steps, longest run of consecutive skips]
        self._acc = torch.zeros(7, device=self.device)
        self._bad_run = torch.zeros((), device=self.device)

    def train_step(self, batch: dict[str, torch.Tensor]) -> None:
        """One optimizer step with no host synchronization: a non-finite step is skipped on the device."""
        self._seed()
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        if self.latents is not None:
            batch["latents"] = self.latents[batch.pop("lat_idx")].float() * batch["fmask"][..., None]
        lr = _lr(self.cfg, self.step) * self.lr_scale
        for o in self.optimizers:
            for g in o.param_groups:
                g["lr"] = lr
        use_bf16 = self.cfg.precision == "bf16" and self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            losses = self.net(**batch, cond_drop=self.cfg.cond_drop, dur_weight=self.cfg.dur_weight)
        losses.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        bad = (~torch.isfinite(grad_norm)).float()  # a NaN/inf loss makes the norm non-finite (same on every rank)
        for o in self.optimizers:
            if isinstance(o, torch.optim.AdamW):
                # Fused AdamW reads these (as set by GradScaler) and skips the update on the device when found_inf=1.
                o.found_inf = bad  # pyright: ignore[reportAttributeAccessIssue]
                o.grad_scale = None  # pyright: ignore[reportAttributeAccessIssue]
                o.step()
            elif not bad.item():  # Muon has no device-side skip; it syncs
                o.step()
            o.zero_grad(set_to_none=True)
        decay = min(self.cfg.ema_decay, (1 + self.step) / (10 + self.step))
        with torch.no_grad():  # a skipped step left the model unchanged, so this average stays valid
            torch._foreach_lerp_(  # pyright: ignore[reportPrivateImportUsage]
                list(self.average.parameters()), list(self.model.parameters()), 1 - decay)
            good = 1 - bad
            values = torch.stack([losses.total.detach(), losses.flow.detach(), losses.duration.detach(),
                                  grad_norm.detach()]).float()
            self._acc[:4] += torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) * good
            self._acc[4] += good
            self._acc[5] += bad
            self._bad_run = (self._bad_run + bad) * bad
            self._acc[6] = torch.maximum(self._acc[6], self._bad_run)
        self._lr = lr

    def run(self) -> None:
        cfg = self.cfg
        _Stop.install()
        self._last_save, self._saved_step = time.monotonic(), self.step
        self._reset_accumulators()
        started, frames_seen = time.monotonic(), 0
        while self.step < cfg.steps and not _Stop.requested and not self.stopped_early:
            for batch in self._batches():
                frames_seen += int(batch["fmask"].sum())  # host-side tensor: no device sync
                self.train_step(batch)
                self.batch_in_epoch += 1
                self.step += 1
                rate = frames_seen * self.world / (time.monotonic() - started)
                if self.step % cfg.log_every == 0 and self._check_and_log(rate):  # every rank: rollbacks must agree
                    break
                if cfg.sample_every and self.step % cfg.sample_every == 0:
                    self._evaluate()
                due = time.monotonic() - self._last_save > cfg.ckpt_every_minutes * 60
                end = _Stop.requested or self.step >= cfg.steps or self.stopped_early
                if self.step % cfg.ckpt_every_steps == 0 or due or end:
                    self._save()
                if end:
                    break
            else:
                self.epoch += 1
                self.batch_in_epoch = 0
        self._save(wait=True)
        if self.main and (self.step >= cfg.steps or self.stopped_early):
            reason = "early stop" if self.stopped_early else "steps"
            (cfg.run_dir / "DONE").write_text(f"{self.step} ({reason}, best CER {self.best_cer:.4f})")
        if self.world > 1:
            dist.destroy_process_group()

    def _check_and_log(self, frames_per_s: float) -> bool:
        """One host sync per log interval: read the accumulated metrics, log them, roll back if training diverged.
        Returns True after a rollback (the epoch's batch iterator must restart from the restored position)."""
        acc = self._acc.tolist()
        self._acc.zero_()
        good, skipped, worst_run = acc[4], int(acc[5]), int(acc[6])
        if skipped:
            log.warning("%d non-finite steps skipped before step %d (longest run %d)", skipped, self.step, worst_run)
        if worst_run >= self.cfg.max_bad_steps:
            self._rollback()
            return True
        if self.main:
            avg = dict(zip(("loss", "flow", "dur", "grad_norm"), (a / good if good else math.nan for a in acc[:4]),
                           strict=True))
            record = {"step": self.step, "epoch": self.epoch, **avg, "lr": self._lr, "skipped": skipped,
                      "audio_s_per_s": frames_per_s / 25, "time": time.time()}
            with open(self.cfg.run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            atomic_write_bytes(self.cfg.run_dir / "heartbeat.json", json.dumps(record).encode())
            log.info("step %d | loss %.4f flow %.4f dur %.4f | grad %.2f lr %.2e | %.0f s audio/s", self.step,
                     avg["loss"], avg["flow"], avg["dur"], avg["grad_norm"], self._lr, record["audio_s_per_s"])
        return False

    def _evaluate(self) -> None:
        """Samples and CER on the main rank; early stopping and the best checkpoint follow the CER."""
        stop = torch.zeros((), device=self.device)
        if self.main:
            cer = self.sample()
            if cer < self.best_cer - self.cfg.early_stop_min_delta:
                self.best_cer, self.stale_evals = cer, 0
                self.ckpt.save(self.step, self._state(), name="best.pt")
                log.info("new best CER %.4f at step %d: saved best.pt", cer, self.step)
            else:
                self.stale_evals += 1
                if self.cfg.early_stop_patience and self.stale_evals >= self.cfg.early_stop_patience:
                    log.info("no CER improvement in %d evaluations: stopping early", self.stale_evals)
                    stop.fill_(1)
        if self.world > 1:
            dist.broadcast(stop, 0)
        self.stopped_early = bool(stop.item())

    def sample(self) -> float:
        """Synthesize the configured sentences with the weight average into samples/, and log intelligibility (CER of
        the Turkish recognizer on held-out sentences)."""
        import soundfile as sf

        from okur.data.align import Recognizer
        from okur.data.codec import RATE, Codec
        from okur.evals.intelligibility import SENTENCES, character_error_rate, greedy
        from okur.synth import Synthesizer

        if not hasattr(self, "_codec"):
            self._codec, self._recognizer = Codec(str(self.device)), Recognizer(str(self.device))
        synth = Synthesizer(self.average, self._codec.decode, str(self.device))
        spk = self.corpus.speakers.index(self.cfg.sample_speaker) if self.cfg.sample_speaker else 0
        out = self.cfg.run_dir / "samples"
        out.mkdir(exist_ok=True)
        for i, text in enumerate(self.cfg.sample_texts):
            audio = synth(text, speaker=spk, steps=self.cfg.sample_steps, guidance=self.cfg.guidance)
            sf.write(out / f"step-{self.step:08d}-{i}.wav", audio, RATE)
        pairs = [(synth.frontend(t), greedy(self._recognizer, synth(t, speaker=spk, steps=self.cfg.sample_steps,
                                                                    guidance=self.cfg.guidance))) for t in SENTENCES]
        cer = character_error_rate(pairs)
        with open(self.cfg.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": self.step, "cer": cer, "example": pairs[0][1], "time": time.time()}) + "\n")
        log.info("step %d | CER %.3f | heard: %s", self.step, cer, pairs[0][1])
        return cer


def main(config: Path) -> None:
    cfg = TrainConfig.load(config)
    Trainer(cfg).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    main(Path(sys.argv[1]))
