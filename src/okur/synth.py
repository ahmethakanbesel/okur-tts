"""Text to speech with a trained acoustic model. Latents are decoded by our decoder when given, else by the codec."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32

from okur.frontend import Frontend, encode
from okur.model.acoustic import Acoustic
from okur.model.timeline import frame_timeline, word_frames, words


class Synthesizer:
    def __init__(self, model: Acoustic, decode: Callable[[torch.Tensor], Float32[np.ndarray, "..."]],
                 device: str) -> None:
        self.model, self.decode, self.device = model.eval(), decode, device
        self.frontend = Frontend()

    @torch.no_grad()
    def latents(self, text: str, *, speaker: int = 0, quality: int = 0, steps: int = 16, guidance: float = 2.0,
                speed: float = 1.0, seed: int = 0) -> torch.Tensor:
        letters = self.frontend(text)
        dev = self.device
        ids = torch.tensor([encode(letters)], device=dev)
        mask = torch.ones_like(ids, dtype=torch.bool)
        out = self.model.text_stage(ids, mask, torch.tensor([speaker], device=dev), torch.tensor([quality], device=dev))
        dur = self.model.frames_from_log(out.log_dur, mask) / speed
        w = words(letters)
        fw, fp = frame_timeline(word_frames(dur[0].cpu(), w.cw, w.n_words))
        cond = self.model.condition(out.h, dur, mask, w.cw[None].to(dev), w.wstart[None].to(dev), fw[None].to(dev),
                                    fp[None].to(dev))
        gen = torch.Generator().manual_seed(seed)
        fmask = torch.ones(1, len(fw), dtype=torch.bool, device=dev)
        times = self.model.cfg.distilled_times
        if times:  # a distilled student: a few steps, guidance already baked in
            noises = torch.randn(len(times), 1, len(fw), self.model.cfg.latent_dim, generator=gen).to(dev)
            return self.model.sample_distilled(cond, out.g, fmask, noises, times)[0]
        noise = torch.randn(1, len(fw), self.model.cfg.latent_dim, generator=gen).to(dev)
        return self.model.sample(cond, out.g, fmask, noise, steps=steps, guidance=guidance)[0]

    def __call__(self, text: str, *, speaker: int = 0, quality: int = 0, steps: int = 16, guidance: float = 2.0,
                 speed: float = 1.0, seed: int = 0) -> Float32[np.ndarray, "..."]:
        return self.decode(self.latents(text, speaker=speaker, quality=quality, steps=steps, guidance=guidance,
                                        speed=speed, seed=seed))


def torch_synthesizer(acoustic: Path, decoder: Path, device: str | None = None) -> Synthesizer:
    """A Synthesizer from a checkpoint or release pair, decoding with our decoder on the best available device."""
    from okur import release
    from okur.model.config import AcousticConfig, DecoderConfig
    from okur.model.decoder import Decoder

    dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    state = release.read(acoustic)
    model = Acoustic(AcousticConfig.model_validate(state["model_config"]))
    model.load_state_dict(state["average"])
    dstate = release.read(decoder)
    dec = Decoder(DecoderConfig.model_validate(dstate["model_config"])).to(dev).eval()
    dec.load_state_dict(dstate["model"])

    @torch.no_grad()
    def decode(z: torch.Tensor) -> Float32[np.ndarray, "..."]:
        return dec(z.T[None].float())[0].float().cpu().numpy()

    return Synthesizer(model.to(dev), decode, dev)
