"""Apple MLX port of the HiFi-GAN-style decoder (latents → 48 kHz audio). Inference only, channels-last layout.

Mirrors okur.model.decoder; tests/test_mlx.py checks they agree on the same weights.
"""

import math
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

from okur.model.config import DecoderConfig


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.convs1 = [nn.Conv1d(ch, ch, k, dilation=d, padding=(k * d - d) // 2) for d in dilations]
        self.convs2 = [nn.Conv1d(ch, ch, k, padding=(k - 1) // 2) for _ in dilations]

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            x = x + c2(nn.leaky_relu(c1(nn.leaky_relu(x, 0.1)), 0.1))
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        ch, rates = cfg.ch, cfg.rates
        self.hop, self.nk = math.prod(rates), len(cfg.rb_kernels)
        self.pre = nn.Conv1d(cfg.latent_dim, ch, 7, padding=3)
        self.ups = [nn.ConvTranspose1d(ch >> i, ch >> (i + 1), k, stride=r, padding=(k - r) // 2)
                    for i, (r, k) in enumerate(zip(rates, cfg.kernels, strict=True))]
        self.blocks = [ResBlock(ch >> (i + 1), k, d) for i in range(len(rates))
                       for k, d in zip(cfg.rb_kernels, cfg.rb_dilations, strict=True)]
        self.post = nn.Conv1d(ch >> len(rates), 1, 7, padding=3)

    def __call__(self, z: mx.array) -> mx.array:
        """z: (b, frames, 64) → (b, frames * hop) audio."""
        frames = z.shape[1]
        x = self.pre(z)
        for i, up in enumerate(self.ups):
            x = up(nn.leaky_relu(x, 0.1))
            blocks = self.blocks[i * self.nk:(i + 1) * self.nk]
            x = sum((b(x) for b in blocks[1:]), blocks[0](x)) / self.nk
        return mx.tanh(self.post(nn.leaky_relu(x, 0.01)))[:, :frames * self.hop, 0]

    @classmethod
    def from_torch_state(cls, cfg: DecoderConfig, state: dict[str, Any]) -> "Decoder":
        """PyTorch layouts: Conv1d (out, in, k) → (out, k, in); ConvTranspose1d (in, out, k) → (out, k, in)."""
        weights: dict[str, mx.array] = {}
        for key, value in state.items():
            a = np.asarray(value.detach().cpu().float().numpy())
            if a.ndim == 3 and key.startswith("ups."):
                a = a.transpose(1, 2, 0)
            elif a.ndim == 3:
                a = a.transpose(0, 2, 1)
            weights[key] = mx.array(a)
        model = cls(cfg)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        return model
