"""The latent codec used for training targets: ACE-Step 1.5's VAE (MIT), frozen. 48 kHz audio ↔ 64-dim latents at 25 Hz.

Only the encoder is needed to make training data. Its decoder is the reference our small decoder learns to match.
"""

from typing import Any

import numpy as np
import torch
from jaxtyping import Float16, Float32

CODEC_REPO = "ACE-Step/Ace-Step1.5"
RATE = 48000
HOP = 1920
LATENT_RATE = RATE // HOP


class Codec:
    def __init__(self, device: str, dtype: torch.dtype = torch.float32) -> None:
        from diffusers import AutoencoderOobleck  # pyright: ignore[reportPrivateImportUsage]

        vae: Any = AutoencoderOobleck.from_pretrained(CODEC_REPO, subfolder="vae")
        self.vae: Any = vae.to(device, dtype).eval()
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def encode(self, batch: list[Float32[np.ndarray, "..."]]) -> list[Float16[np.ndarray, "..."]]:
        """Mono 48 kHz clips, each a multiple of HOP samples long, to (frames, 64) latents."""
        frames = [len(a) // HOP for a in batch]
        x = torch.zeros(len(batch), 2, max(frames) * HOP, dtype=self.dtype)
        for i, a in enumerate(batch):
            x[i, :, :frames[i] * HOP] = torch.from_numpy(a[:frames[i] * HOP])
        z = self.vae.encode(x.to(self.device)).latent_dist.mean.float().cpu()  # (b, 64, t)
        return [z[i, :, :n].T.numpy().astype(np.float16) for i, n in enumerate(frames)]

    @torch.inference_mode()
    def decode(self, z: Float32[np.ndarray, "..."] | torch.Tensor) -> Float32[np.ndarray, "..."]:
        """(frames, 64) latents to mono 48 kHz audio."""
        zt = torch.as_tensor(z, dtype=self.dtype, device=self.device).T[None]
        return self.vae.decode(zt).sample.mean(1)[0].float().cpu().numpy()
