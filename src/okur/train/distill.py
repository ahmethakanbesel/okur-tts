"""DMD2 distillation: a 4-step student from the 16-step teacher with guidance baked in (Yin et al., 2024).

Three copies of the acoustic model:
- teacher (frozen): the target distribution, sampled with classifier-free guidance;
- student (trained): predicts clean latents in `times` steps, re-noising between steps, no guidance at inference;
- fake (trained): tracks the student's current distribution with an ordinary flow-matching loss.

Distribution matching: for a student sample x1, noise it to x_t; the teacher's and the fake's clean-latent predictions
at x_t differ where the student's distribution differs from the teacher's. The student is moved toward the teacher's
prediction and away from the fake's: grad = (p_fake - p_real) / mean|x1 - p_real|, loss = ½‖x1 - sg(x1 - grad)‖².
Two-time-scale: `fake_ratio` fake updates per student update. Data only supplies conditions (text, durations,
speaker); the targets come from the teacher, so no audio is regressed.

The text encoder, duration predictor and aligner stay the teacher's: only the latent generator changes.
Built on the acoustic Trainer, so checkpointing, resume, signal handling and the supervisor work unchanged.
"""

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from okur.data.shards import atomic_write_bytes
from okur.model.acoustic import Acoustic
from okur.model.config import AcousticConfig
from okur.train.config import TrainConfig
from okur.train.trainer import Trainer, _lr

log = logging.getLogger(__name__)


class DistillConfig(TrainConfig):
    teacher: Path  # checkpoint of the (fine-tuned) teacher; the student starts from its averaged weights
    times: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    real_guidance: float = 3.0
    fake_ratio: int = 5  # fake-model updates per student update
    fake_lr: float | None = None  # defaults to lr

    @classmethod
    def load(cls, path: Path) -> "DistillConfig":
        return cls.model_validate(yaml.safe_load(path.read_text()))


def _masked_mse(a: Tensor, b: Tensor, fmask: Tensor) -> Tensor:
    m = fmask.float()
    return (((a - b) ** 2).mean(-1) * m).sum() / m.sum()


class DistillTrainer(Trainer):
    cfg: DistillConfig

    def __init__(self, cfg: DistillConfig) -> None:
        super().__init__(cfg.model_copy(update={"init_from": cfg.teacher}))
        self.cfg = cfg
        times = tuple(cfg.times)
        for m in (self.model, self.average):  # the student samples in `times` steps from now on
            m.cfg = m.cfg.model_copy(update={"distilled_times": times})
        teacher_state = torch.load(cfg.teacher, map_location="cpu", weights_only=False)
        self.teacher = Acoustic(AcousticConfig.model_validate(teacher_state["model_config"])
                                .model_copy(update={"n_speakers": len(self.corpus.speakers)})).to(self.device)
        self.teacher.load_state_dict(teacher_state["average"])
        self.teacher.eval().requires_grad_(False)
        self.fake = copy.deepcopy(self.teacher).train().requires_grad_(True)
        self.fake_opt = torch.optim.AdamW(self.fake.parameters(), lr=cfg.fake_lr or cfg.lr, betas=(0.9, 0.98),
                                          weight_decay=0.0, fused=True)
        latest = self.ckpt.load_latest()
        if latest is not None and "fake" in latest:
            self.fake.load_state_dict(latest["fake"])
            self.fake_opt.load_state_dict(latest["fake_opt"])

    def _state(self) -> dict[str, Any]:
        return super()._state() | {"fake": self.fake.state_dict(), "fake_opt": self.fake_opt.state_dict()}

    def _restore(self, state: dict[str, Any]) -> None:
        super()._restore(state)
        if hasattr(self, "fake") and "fake" in state:
            self.fake.load_state_dict(state["fake"])
            self.fake_opt.load_state_dict(state["fake_opt"])

    # ---- one step ---------------------------------------------------------------------------------------------------

    def _student(self, cond: Tensor, g: Tensor, fmask: Tensor, grad_at: int | None) -> Tensor:
        """A student sample. With grad_at=k, steps before k run without gradient and step k with it (backward
        simulation: train every step on its own input distribution)."""
        times = self.cfg.times
        last = len(times) - 1 if grad_at is None else grad_at
        noises = torch.randn(last + 1, *cond.shape[:2], self.model.cfg.latent_dim, device=cond.device)
        x = noises[0]
        with torch.no_grad():
            for j in range(last):
                x1 = self.model.student_x1(x, cond, g, fmask, times[j])
                x = (1 - times[j + 1]) * noises[j + 1] + times[j + 1] * x1
        if grad_at is None:
            with torch.no_grad():
                return self.model.student_x1(x, cond, g, fmask, times[last]) * fmask[..., None]
        return self.model.student_x1(x, cond, g, fmask, times[last]) * fmask[..., None]

    def _noised(self, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        t = torch.sigmoid(torch.randn(x1.shape[0], device=x1.device)).clamp(0.02, 0.98)
        e = torch.randn_like(x1)
        return (1 - t)[:, None, None] * e + t[:, None, None] * x1, t, e

    def _student_loss(self, cond: Tensor, g: Tensor, fmask: Tensor) -> Tensor:
        k = int(torch.randint(len(self.cfg.times), ()).item())
        x1 = self._student(cond, g, fmask, grad_at=k)
        xt, t, _ = self._noised(x1.detach())
        with torch.no_grad():
            both = self.teacher.velocity(torch.cat([xt, xt]), torch.cat([cond, torch.zeros_like(cond)]),
                                         torch.cat([t, t]), torch.cat([g, g]), torch.cat([fmask, fmask]))
            vc, vu = both.chunk(2)
            w = self.cfg.real_guidance
            one_minus_t = (1 - t)[:, None, None]
            p_real = xt + one_minus_t * (vu + w * (vc - vu))
            p_fake = xt + one_minus_t * self.fake.velocity(xt, cond, t, g, fmask)
            m = fmask[..., None].float()
            scale = ((x1.detach() - p_real).abs() * m).sum((1, 2), keepdim=True) / (m.sum((1, 2), keepdim=True) * 64)
            direction = torch.nan_to_num((p_fake - p_real) / scale.clamp(min=1e-5))
        return 0.5 * _masked_mse(x1, (x1 - direction).detach(), fmask)

    def _fake_loss(self, cond: Tensor, g: Tensor, fmask: Tensor) -> Tensor:
        x1 = self._student(cond, g, fmask, grad_at=None)
        xt, t, e = self._noised(x1)
        return _masked_mse(self.fake.velocity(xt, cond, t, g, fmask), x1 - e, fmask)

    def train_step(self, batch: dict[str, Tensor]) -> None:
        self._seed()
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        lr = _lr(self.cfg, self.step) * self.lr_scale
        for o in (*self.optimizers, self.fake_opt):
            for group in o.param_groups:
                group["lr"] = lr if o is not self.fake_opt else (self.cfg.fake_lr or lr)
        bf16 = self.cfg.precision == "bf16" and self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
            with torch.no_grad():
                text = self.teacher.text_stage(batch["ids"], batch["mask"], batch["speaker"], batch["quality"])
                cond = self.teacher.condition(text.h, batch["dur"], batch["mask"], batch["cw"], batch["wstart"],
                                              batch["fw"], batch["fp"])
            student_turn = self.step % (self.cfg.fake_ratio + 1) == self.cfg.fake_ratio
            if student_turn:
                loss = self._student_loss(cond, text.g, batch["fmask"])
                opt, params = self.optimizers[0], list(self.model.parameters())
            else:
                loss = self._fake_loss(cond, text.g, batch["fmask"])
                opt, params = self.fake_opt, list(self.fake.parameters())
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        bad = (~torch.isfinite(norm)).float()
        opt.found_inf = bad  # pyright: ignore[reportAttributeAccessIssue]
        opt.grad_scale = None  # pyright: ignore[reportAttributeAccessIssue]
        opt.step()
        with torch.no_grad():
            if student_turn:
                decay = min(self.cfg.ema_decay, (1 + self.step) / (10 + self.step))
                torch._foreach_lerp_(  # pyright: ignore[reportPrivateImportUsage]
                    list(self.average.parameters()), list(self.model.parameters()), 1 - decay)
            good = 1 - bad
            value = torch.nan_to_num(loss.detach().float()) * good
            slot = 1 if student_turn else 2  # [_, student loss, fake loss, grad norm, good, skipped, longest skip run]
            self._acc[slot] += value
            self._acc[3] += torch.nan_to_num(norm.detach().float()) * good
            self._acc[4] += good
            self._acc[5] += bad
            self._bad_run = (self._bad_run + bad) * bad
            self._acc[6] = torch.maximum(self._acc[6], self._bad_run)
        self._lr = lr

    def _check_and_log(self, frames_per_s: float) -> bool:
        acc = self._acc.tolist()
        self._acc.zero_()
        good, skipped, worst = acc[4], int(acc[5]), int(acc[6])
        if worst >= self.cfg.max_bad_steps:
            self._rollback()
            return True
        if self.main and good:
            n_student = max(1.0, good / (self.cfg.fake_ratio + 1))
            record = {"step": self.step, "student": acc[1] / n_student, "fake": acc[2] / max(1.0, good - n_student),
                      "grad_norm": acc[3] / good, "lr": self._lr, "skipped": skipped,
                      "audio_s_per_s": frames_per_s / 25, "time": time.time()}
            with open(self.cfg.run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
            atomic_write_bytes(self.cfg.run_dir / "heartbeat.json", json.dumps(record).encode())
            log.info("step %d | student %.4f fake %.4f | grad %.2f lr %.2e | %.0f s audio/s", self.step,
                     record["student"], record["fake"], record["grad_norm"], self._lr, record["audio_s_per_s"])
        return False


def main(config: Path) -> None:
    DistillTrainer(DistillConfig.load(config)).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    main(Path(sys.argv[1]))
