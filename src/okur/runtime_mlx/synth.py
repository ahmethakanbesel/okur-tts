"""Text to latents on Apple Silicon with MLX, from a training checkpoint's averaged weights."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from okur import release
from okur.frontend import Frontend, encode
from okur.model.config import AcousticConfig, DecoderConfig
from okur.model.timeline import frame_timeline, word_frames, words
from okur.runtime_mlx.acoustic import Acoustic
from okur.runtime_mlx.decoder import Decoder


class MlxSynthesizer:
    def __init__(self, checkpoint: Path, decoder: Path | None = None) -> None:
        state = release.read(checkpoint)
        self.decoder: Decoder | None = None
        if decoder is not None:
            dstate = release.read(decoder)
            self.decoder = Decoder.from_torch_state(DecoderConfig.model_validate(dstate["model_config"]),
                                                    dstate["model"])
        self.cfg = AcousticConfig.model_validate(state["model_config"])
        self.model = Acoustic.from_torch_state(self.cfg, state["average"])
        self.speakers: list[str] = state["speakers"]
        self.frontend = Frontend()

    def latents(self, text: str, *, speaker: int = 0, quality: int = 0, steps: int = 16, guidance: float = 2.0,
                speed: float = 1.0, seed: int = 0) -> np.ndarray:
        letters = self.frontend(text)
        ids = mx.array([encode(letters)])
        mask = mx.ones(ids.shape, dtype=mx.bool_)
        h, log_dur, g = self.model.text_stage(ids, mask, mx.array([speaker]), mx.array([quality]))
        dur = self.model.frames_from_log(log_dur, mask) / speed
        w = words(letters)
        fw, fp = frame_timeline(word_frames(torch.from_numpy(np.array(dur[0])), w.cw, w.n_words))
        cond = self.model.condition(h, dur, mask, mx.array(w.cw[None].numpy()), mx.array(w.wstart[None].numpy()),
                                    mx.array(fw[None].numpy()), mx.array(fp[None].numpy()))
        gen = torch.Generator().manual_seed(seed)  # same noise as the PyTorch path
        fmask = mx.ones((1, len(fw)), dtype=mx.bool_)
        times = self.cfg.distilled_times
        if times:
            noises = mx.array(torch.randn(len(times), 1, len(fw), self.cfg.latent_dim, generator=gen).numpy())
            z = self.model.sample_distilled(cond, g, fmask, noises, times)
        else:
            noise = mx.array(torch.randn(1, len(fw), self.cfg.latent_dim, generator=gen).numpy())
            z = self.model.sample(cond, g, fmask, noise, steps=steps, guidance=guidance)
        return np.array(z[0])

    def say(self, text: str, **kwargs: float) -> np.ndarray:
        """Text to 48 kHz audio, entirely in MLX (needs the decoder checkpoint)."""
        if self.decoder is None:
            raise ValueError("MlxSynthesizer was created without a decoder checkpoint")
        z = self.latents(text, **kwargs)  # type: ignore[arg-type]
        audio = self.decoder(mx.array(z[None]))
        return np.array(audio[0])
