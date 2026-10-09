"""The decoder GAN trains on prepared shards, crosses into adversarial training, and resumes from its checkpoint."""

import io
from pathlib import Path

import numpy as np
import soundfile as sf

from okur.data.codec import HOP, RATE
from okur.data.shards import Example, write_shard
from okur.model.config import DecoderConfig
from okur.train.decoder_trainer import DecoderTrainConfig, DecoderTrainer


def test_decoder_gan_trains_and_resumes(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    examples = []
    for i in range(6):
        frames = 40
        buf = io.BytesIO()
        sf.write(buf, (0.1 * rng.standard_normal(frames * HOP)).astype(np.float32), RATE, format="FLAC")
        examples.append(Example(f"x{i}", "synthetic", "s", 0, "a", "a", np.array([frames], np.float32),
                                rng.standard_normal((frames, 64)).astype(np.float16), 0.0, buf.getvalue()))
    (tmp_path / "data").mkdir()
    write_shard(tmp_path / "data", 0, examples, {})

    def cfg(steps: int) -> DecoderTrainConfig:
        return DecoderTrainConfig(run_dir=tmp_path / "run", data=[tmp_path / "data"], steps=steps, batch=2,
                                  segment_frames=8, mel_only_steps=2, workers=0, device="cpu", ckpt_every_steps=2,
                                  log_every=1, model=DecoderConfig(ch=64), val_clips=2, sample_every=2)

    DecoderTrainer(cfg(3)).run()
    resumed = DecoderTrainer(cfg(5))
    assert resumed.step == 3
    resumed.run()
    assert (tmp_path / "run" / "DONE").read_text() == "5"
    assert len(list((tmp_path / "run" / "samples").glob("step-00000004-*.wav"))) == 2  # held-out clips decoded
