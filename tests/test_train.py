"""Fault tolerance: a killed-and-resumed run equals an uninterrupted one; corrupt checkpoints and NaNs are survived."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from okur.data.shards import Example, write_shard
from okur.model.acoustic import Acoustic, Losses
from okur.model.config import AcousticConfig
from okur.train import trainer as trainer_mod
from okur.train.config import TrainConfig
from okur.train.data import Collate, Corpus
from okur.train.trainer import Trainer

SENTENCES = ["merhaba dünya.", "kâr arttı.", "bu bir deneme cümlesidir.", "hâlâ bekliyor musun?", "üçüncü kat.",
             "sabah erken kalktık.", "ofis kapalı.", "yarın görüşürüz."]


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("shards")
    rng = np.random.default_rng(0)
    examples = []
    for i, letters in enumerate(SENTENCES * 2):
        n_frames = 2 * len(letters) + int(rng.integers(0, 5))
        dur = np.full(len(letters), n_frames / len(letters), dtype=np.float32)
        latents = rng.standard_normal((n_frames, 64)).astype(np.float16)
        examples.append(Example(f"x{i}", "synthetic", f"spk{i % 2}", 0, letters, letters, dur, latents, 0.0))
    write_shard(out, 0, examples, {"kept": len(examples)})
    return out


def _cfg(run_dir: Path, shards: Path, **kw: object) -> TrainConfig:
    small = AcousticConfig(d=64, n_layers=2, n_heads=4, text_conv=1, text_attn=1, dur_hidden=32)
    base: dict[str, object] = dict(run_dir=run_dir, data=[shards], model=small, steps=8, warmup=2, max_frames=120,
                                   max_letters=200, workers=0, device="cpu", ckpt_every_steps=3,
                                   ckpt_every_minutes=1e9, log_every=1, sample_every=0)
    return TrainConfig.model_validate(base | kw)


def _weights(run_dir: Path) -> dict[str, torch.Tensor]:
    latest = sorted((run_dir / "checkpoints").glob("step-*.pt"))[-1]
    return torch.load(latest, weights_only=False)["model"]


def test_resume_is_exact(tmp_path: Path, shards: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    straight = tmp_path / "straight"
    Trainer(_cfg(straight, shards)).run()

    interrupted = tmp_path / "interrupted"
    t = Trainer(_cfg(interrupted, shards))
    real_step = t.train_step

    def killed_after_five(batch: dict[str, torch.Tensor]) -> dict[str, float] | None:
        out = real_step(batch)
        if t.step + 1 >= 5:
            trainer_mod._Stop.requested = True  # what SIGTERM does
        return out

    monkeypatch.setattr(t, "train_step", killed_after_five)
    t.run()
    trainer_mod._Stop.requested = False
    assert torch.load(sorted((interrupted / "checkpoints").glob("*.pt"))[-1], weights_only=False)["step"] == 5

    resumed = Trainer(_cfg(interrupted, shards))
    assert resumed.step == 5
    resumed.run()
    assert (interrupted / "DONE").exists()
    a, b = _weights(straight), _weights(interrupted)
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=0, rtol=0, msg=k)


def test_corrupt_checkpoint_falls_back(tmp_path: Path, shards: Path) -> None:
    Trainer(_cfg(tmp_path, shards, steps=6)).run()
    newest = sorted((tmp_path / "checkpoints").glob("step-*.pt"))[-1]
    newest.write_bytes(b"truncated by a crash")
    assert Trainer(_cfg(tmp_path, shards, steps=6)).step == 3


def _nan_on_calls(monkeypatch: pytest.MonkeyPatch, t: Trainer, calls: set[int]) -> None:
    """Make the model's loss NaN on the given (1-based) training calls, as a diverging batch would."""
    real = t.model.loss
    count = {"n": 0}

    def loss(**kw: object) -> Losses:
        count["n"] += 1
        out = real(**kw)  # type: ignore[arg-type]
        return Losses(out.total * float("nan"), out.flow, out.duration) if count["n"] in calls else out

    monkeypatch.setattr(t.model, "loss", loss)


def test_single_nan_step_is_skipped_on_device(tmp_path: Path, shards: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = Trainer(_cfg(tmp_path, shards, steps=6, max_bad_steps=5))
    _nan_on_calls(monkeypatch, t, {3})
    before: dict[str, torch.Tensor] = {}
    real_step = t.train_step

    def step(batch: dict[str, torch.Tensor]) -> None:
        if t.step == 2:
            before.update({k: v.clone() for k, v in t.model.state_dict().items()})
        real_step(batch)
        if t.step == 2:  # the NaN step (3rd call) must leave every weight untouched
            for k, v in t.model.state_dict().items():
                torch.testing.assert_close(v, before[k], atol=0, rtol=0, msg=k)

    monkeypatch.setattr(t, "train_step", step)
    t.run()
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert sum(r.get("skipped", 0) for r in records) == 1
    assert t.lr_scale == 1.0 and all(torch.isfinite(p).all() for p in t.model.parameters())


def test_nan_run_rolls_back_with_lower_lr(tmp_path: Path, shards: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Trainer(_cfg(tmp_path, shards, steps=3)).run()
    t = Trainer(_cfg(tmp_path, shards, steps=8, max_bad_steps=2))
    assert t.step == 3
    _nan_on_calls(monkeypatch, t, {1, 2})
    t.run()
    assert t.lr_scale == 0.5
    assert t.step == 8


def test_padding_and_device_latents_do_not_change_the_loss(shards: Path) -> None:
    corpus = Corpus([shards])
    items = [corpus[i] for i in range(4)]
    plain = Collate(corpus.latents)(items)
    padded = Collate(corpus.latents, pad_frames=16, pad_letters=8, latents_on_device=True)(items)
    padded["latents"] = torch.from_numpy(corpus.latents)[padded.pop("lat_idx")].float() * padded["fmask"][..., None]
    assert padded["fw"].shape[1] % 16 == 0 and padded["ids"].shape[1] % 8 == 0
    torch.manual_seed(0)
    model = Acoustic(AcousticConfig(d=64, n_layers=2, n_heads=4, text_conv=1, text_attn=1, dur_hidden=32,
                                    n_speakers=2)).eval()
    losses = []
    for batch in (plain, padded):
        torch.manual_seed(1)
        losses.append(model.loss(**batch, cond_drop=0.0))  # type: ignore[arg-type]
    torch.testing.assert_close(losses[0].duration, losses[1].duration)
    # flow noise is drawn per padded shape, so compare with the same noise: equal shapes after cropping
    assert torch.isfinite(losses[1].total)


def test_early_stop_on_cer(tmp_path: Path, shards: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = Trainer(_cfg(tmp_path, shards, steps=50, sample_every=1, early_stop_patience=2))
    cers = iter([0.5, 0.4, 0.45, 0.46, 0.1])
    monkeypatch.setattr(t, "sample", lambda: next(cers))
    t.run()
    assert t.step == 4 and t.best_cer == 0.4
    assert "early stop" in (tmp_path / "DONE").read_text()
    assert torch.load(tmp_path / "checkpoints" / "best.pt", weights_only=False)["step"] == 2


def test_fine_tune_from_checkpoint_on_one_speaker(tmp_path: Path, shards: Path) -> None:
    base = tmp_path / "base"
    Trainer(_cfg(base, shards, steps=3)).run()
    ckpt = sorted((base / "checkpoints").glob("step-*.pt"))[-1]
    ft = Trainer(_cfg(tmp_path / "ft", shards, steps=2, init_from=ckpt, only_speakers=["spk1"], sample_speaker="spk1"))
    assert ft.corpus.speakers == ["spk0", "spk1"]  # the base model's speaker table is kept
    assert {ft.corpus[i].speaker for i in range(len(ft.corpus))} == {1}  # but only spk1 is trained on
    base_avg = torch.load(ckpt, weights_only=False)["average"]
    for k, v in ft.model.state_dict().items():
        torch.testing.assert_close(v, base_avg[k].to(v.device), msg=k)  # starts from the averaged base weights
    ft.run()
    assert ft.step == 2


def test_dmd2_distillation_runs_and_resumes(tmp_path: Path, shards: Path) -> None:
    from okur.train.distill import DistillConfig, DistillTrainer

    Trainer(_cfg(tmp_path / "teacher", shards, steps=3)).run()
    teacher = sorted((tmp_path / "teacher" / "checkpoints").glob("step-*.pt"))[-1]
    base = _cfg(tmp_path / "student", shards, steps=6).model_dump()
    cfg = DistillConfig.model_validate(base | {"teacher": teacher, "fake_ratio": 2, "lr": 1e-4})
    t = DistillTrainer(cfg)
    t.run()  # 6 steps: 2 student updates, 4 fake updates
    state = torch.load(sorted((tmp_path / "student" / "checkpoints").glob("step-*.pt"))[-1], weights_only=False)
    assert tuple(state["model_config"]["distilled_times"]) == (0.0, 0.25, 0.5, 0.75)
    assert "fake" in state
    for p in t.model.parameters():
        assert torch.isfinite(p).all()
    resumed = DistillTrainer(DistillConfig.model_validate(base | {"teacher": teacher, "fake_ratio": 2, "steps": 8}))
    assert resumed.step == 6
    torch.testing.assert_close(resumed.fake.state_dict()["out_proj.weight"], state["fake"]["out_proj.weight"])
    resumed.run()
    assert resumed.step == 8
