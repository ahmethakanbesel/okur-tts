"""HiFi-GAN-style decoder: 64-dim latents at 25 Hz in, 48 kHz audio out. Ported from EMA Lightning (Apache 2.0).

With `lengths`, everything past each row's real length is held at zero after every layer, so a window padded to a fixed
shape decodes exactly like the same window unpadded.
"""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from okur.model.config import DecoderConfig


def _keep(x: Tensor, length: Tensor | None) -> Tensor:
    return x if length is None else x * (torch.arange(x.shape[-1], device=x.device) < length[:, None])[:, None]


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(nn.Conv1d(ch, ch, k, dilation=d, padding=(k * d - d) // 2) for d in dilations)
        self.convs2 = nn.ModuleList(nn.Conv1d(ch, ch, k, padding=(k - 1) // 2) for _ in dilations)

    def forward(self, x: Tensor, length: Tensor | None = None) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            y = _keep(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1), length)
            x = _keep(x + c2(y), length)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        ch, rates = cfg.ch, cfg.rates
        self.cfg, self.hop, self.nk = cfg, math.prod(rates), len(cfg.rb_kernels)
        self.pre = nn.Conv1d(cfg.latent_dim, ch, 7, padding=3)
        self.ups = nn.ModuleList(nn.ConvTranspose1d(ch >> i, ch >> (i + 1), k, r, padding=(k - r) // 2)
                                 for i, (r, k) in enumerate(zip(rates, cfg.kernels, strict=True)))
        self.blocks = nn.ModuleList(ResBlock(ch >> (i + 1), k, d) for i in range(len(rates))
                                    for k, d in zip(cfg.rb_kernels, cfg.rb_dilations, strict=True))
        self.post = nn.Conv1d(ch >> len(rates), 1, 7, padding=3)

    def forward(self, z: Float[Tensor, "b c t"], lengths: Tensor | None = None) -> Float[Tensor, "b samples"]:
        frames = z.shape[-1]
        length = lengths
        x = _keep(self.pre(_keep(z, length)), length)
        blocks: list[ResBlock] = list(self.blocks)  # type: ignore[arg-type]
        for i, up in enumerate(self.ups):
            assert isinstance(up, nn.ConvTranspose1d)
            x = up(F.leaky_relu(x, 0.1))
            if length is not None:
                length = (length - 1) * up.stride[0] - 2 * int(up.padding[0]) + up.kernel_size[0]
            x = _keep(x, length)
            x = sum((b(x, length) for b in blocks[i * self.nk:(i + 1) * self.nk]), torch.zeros_like(x)) / self.nk
        return torch.tanh(self.post(F.leaky_relu(x)))[:, 0, :frames * self.hop]
