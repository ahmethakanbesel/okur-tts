"""HiFi-GAN discriminators (multi-period and multi-scale) for training the decoder. Training only; never shipped."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

Features = list[Tensor]


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        chans = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chans[i], chans[i + 1], (5, 1), (3 if i < 4 else 1, 1), padding=(2, 0)))
            for i in range(5))
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, x: Tensor) -> tuple[Tensor, Features]:
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        x = x.view(b, c, -1, self.period)
        feats: Features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class ScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(weight_norm(c) for c in (
            nn.Conv1d(1, 128, 15, 1, padding=7), nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20), nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20), nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2)))
        self.post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: Tensor) -> tuple[Tensor, Features]:
        feats: Features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class Discriminators(nn.Module):
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11), scales: int = 3) -> None:
        super().__init__()
        self.subs = nn.ModuleList([*(PeriodDiscriminator(p) for p in periods),
                                   *(ScaleDiscriminator() for _ in range(scales))])
        self.n_periods = len(periods)

    def forward(self, audio: Tensor) -> list[tuple[Tensor, Features]]:
        x = audio[:, None]
        out = []
        for i, d in enumerate(self.subs):
            scale = i - self.n_periods
            y = F.avg_pool1d(x, 2 ** scale * 2, 2 ** scale, padding=2 ** scale) if scale > 0 else x
            out.append(d(y))
        return out


def mel_loss(fake: Tensor, real: Tensor, rate: int = 48000) -> Tensor:
    """Multi-resolution log-mel L1."""
    loss = fake.new_zeros(())
    for n_fft, hop, n_mels in ((512, 128, 64), (1024, 256, 100), (2048, 512, 128)):
        fb = _mel_filters(n_fft, n_mels, rate, fake.device)
        loss = loss + F.l1_loss(_log_mel(fake, n_fft, hop, fb), _log_mel(real, n_fft, hop, fb))
    return loss / 3


def _log_mel(x: Tensor, n_fft: int, hop: int, fb: Tensor) -> Tensor:
    window = torch.hann_window(n_fft, device=x.device)
    return torch.log(torch.clamp(fb @ torch.stft(x, n_fft, hop, window=window, return_complex=True).abs(), min=1e-5))


_FILTERS: dict[tuple[int, int, int, str], Tensor] = {}


def _mel_filters(n_fft: int, n_mels: int, rate: int, device: torch.device) -> Tensor:
    key = (n_fft, n_mels, rate, str(device))
    if key not in _FILTERS:
        def hz_to_mel(f: Tensor) -> Tensor:
            return 2595 * torch.log10(1 + f / 700)
        top = float(hz_to_mel(torch.tensor(rate / 2)))
        mels = torch.linspace(0.0, top, n_mels + 2)
        hz = 700 * (10 ** (mels / 2595) - 1)
        freqs = torch.linspace(0, rate / 2, n_fft // 2 + 1)
        lower, centre, upper = hz[:-2, None], hz[1:-1, None], hz[2:, None]
        fb = torch.clamp(torch.minimum((freqs - lower) / (centre - lower), (upper - freqs) / (upper - centre)), min=0)
        _FILTERS[key] = fb.to(device)
    return _FILTERS[key]


def gan_losses(real: list[tuple[Tensor, Features]], fake: list[tuple[Tensor, Features]]) -> tuple[Tensor, Tensor]:
    """(discriminator loss, generator adversarial + feature-matching loss), least-squares GAN."""
    zero = real[0][0].new_zeros(())
    d_loss = sum((((1 - r) ** 2).mean() + (f ** 2).mean() for (r, _), (f, _) in zip(real, fake, strict=True)), zero)
    adv = sum((((1 - f) ** 2).mean() for f, _ in fake), zero)
    fm = sum((F.l1_loss(ff, rf.detach()) for (_, rfs), (_, ffs) in zip(real, fake, strict=True)
              for rf, ff in zip(rfs, ffs, strict=True)), zero)
    return d_loss, adv + 2.0 * fm
