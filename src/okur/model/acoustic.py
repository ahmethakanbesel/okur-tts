"""The acoustic model: letters in, 64-dim latents at 25 Hz out.

Architecture follows EMA Lightning (Apache 2.0): ConvNeXt text encoder, letter durations, a windowed Gaussian aligner
on a word–letter timeline, and a DiT latent generator with one modulation projection shared by every block.
Changes: speaker and recording-quality conditioning, optional text attention, durations predicted in log space, and the
flow-matching training loss, which EMA does not publish.

Flow matching convention: t = 0 is noise, t = 1 is data, x_t = (1 - t) x0 + t x1, and the network predicts x1 - x0.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from okur.model.config import AcousticConfig
from okur.model.layers import Attention, ConvNeXtBlock, SwiGLU, TextBlock, apply_rope, masked_group_norm, rope
from okur.model.timeline import letter_positions

MAX_LOG_DUR = 6.0


class TextEncoder(nn.Module):
    def __init__(self, cfg: AcousticConfig) -> None:
        super().__init__()
        self.dh = cfg.d // cfg.n_heads
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d, padding_idx=0)
        self.conv = nn.ModuleList(ConvNeXtBlock(cfg.d) for _ in range(cfg.text_conv))
        self.attn = nn.ModuleList(TextBlock(cfg.d, cfg.n_heads) for _ in range(cfg.text_attn))
        self.norm = nn.LayerNorm(cfg.d)

    def forward(self, ids: Int[Tensor, "b l"], mask: Bool[Tensor, "b l"]) -> Float[Tensor, "b l d"]:
        x = self.emb(ids)
        for block in self.conv:
            x = block(x, mask)
        if len(self.attn):
            cos, sin = rope(torch.arange(ids.shape[1], device=ids.device), self.dh)
            for block in self.attn:
                x = block(x, cos, sin, mask)
        return self.norm(x)


class Duration(nn.Module):
    """log(1 + frames) for every letter."""

    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.c1, self.n1 = nn.Conv1d(d, hidden, 3, padding=1), nn.GroupNorm(1, hidden)
        self.c2, self.n2 = nn.Conv1d(hidden, hidden, 3, padding=1), nn.GroupNorm(1, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, h: Float[Tensor, "b l d"], mask: Bool[Tensor, "b l"]) -> Float[Tensor, "b l"]:
        m = mask[:, None].float()
        x = masked_group_norm(F.silu(self.c1(h.transpose(1, 2) * m)), self.n1, m) * m
        x = masked_group_norm(F.silu(self.c2(x)), self.n2, m)
        return self.out(x.transpose(1, 2)).squeeze(-1).masked_fill(~mask, 0.0)


class Aligner(nn.Module):
    """Each frame attends to the letters of its own word and its neighbours, biased toward its own position."""

    def __init__(self, cfg: AcousticConfig) -> None:
        super().__init__()
        d = cfg.d
        self.h, self.dh = cfg.align_heads, d // cfg.align_heads
        self.pos_scale, self.lookback, self.lookahead = cfg.pos_scale, cfg.lookback, cfg.lookahead
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.frame_q = nn.Parameter(torch.zeros(1, 1, d))
        self.log_sigma = nn.Parameter(torch.zeros(self.h))
        self.bias_w = nn.Parameter(torch.ones(self.h))
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, h: Float[Tensor, "b l d"], cw: Int[Tensor, "b l"], cp: Float[Tensor, "b l"],
                fw: Int[Tensor, "b t"], fp: Float[Tensor, "b t"], mask: Bool[Tensor, "b l"]) -> Float[Tensor, "b t d"]:
        b, n_letters, d = h.shape
        t = fw.shape[1]
        # Positions in letter units: word offset in letters plus the position scaled by the word's letter count.
        c, f = cw.clamp(min=0), fw.clamp(min=0)
        wlen = torch.zeros(b, n_letters, device=h.device).scatter_add_(1, c, (cw >= 0).float()).clamp(min=1.0)
        woff = wlen.cumsum(-1) - wlen
        cg = woff.gather(1, c) + cp * wlen.gather(1, c)
        fg = woff.gather(1, f) + fp * wlen.gather(1, f)
        q = self.q(self.frame_q.expand(b, t, d)).view(b, t, self.h, self.dh).transpose(1, 2)
        k = self.k(h).view(b, n_letters, self.h, self.dh).transpose(1, 2)
        v = self.v(h).view(b, n_letters, self.h, self.dh).transpose(1, 2)
        q = apply_rope(q, *(x[:, None] for x in rope(fg * self.pos_scale, self.dh)))
        k = apply_rope(k, *(x[:, None] for x in rope(cg * self.pos_scale, self.dh)))
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh) * self.log_temp.exp()
        sig2 = (self.log_sigma.exp() ** 2).view(1, self.h, 1, 1)
        dist2 = ((fg[:, :, None] - cg[:, None, :]) ** 2)[:, None]
        logits = logits - self.bias_w.view(1, self.h, 1, 1) * dist2 / (2 * sig2)
        rel = cw[:, None, :] - fw[:, :, None]
        allow = (rel >= -self.lookback) & (rel <= self.lookahead) & mask[:, None, :] & (cw[:, None, :] >= 0)
        attn = logits.masked_fill(~allow[:, None], -1e4).softmax(-1) * allow[:, None]
        return self.o((attn @ v).transpose(1, 2).reshape(b, t, d))


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: float) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, heads)
        self.ff = SwiGLU(d, ff_mult)
        self.ada_offset = nn.Parameter(torch.zeros(6, d))

    def forward(self, x: Tensor, mod: Float[Tensor, "b 6 d"], cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        sa, ga, aa, sf, gf, af = (y.unsqueeze(1) for y in (mod + self.ada_offset[None]).unbind(1))
        x = x + aa * self.attn(self.n1(x) * (1 + ga) + sa, cos, sin, mask)
        return x + af * self.ff(self.n2(x) * (1 + gf) + sf)



class TimestepEmbed(nn.Module):
    def __init__(self, d: int, freq: int = 256) -> None:
        super().__init__()
        self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t: Float[Tensor, " b"]) -> Float[Tensor, "b d"]:
        half = self.freq // 2
        a = t[:, None] * 1000.0 * torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)[None]
        return self.mlp(torch.cat([a.cos(), a.sin()], -1))


class TextOut(NamedTuple):
    h: Float[Tensor, "b l d"]
    log_dur: Float[Tensor, "b l"]
    g: Float[Tensor, "b d"]  # global condition: speaker + quality


class Losses(NamedTuple):
    total: Tensor
    flow: Tensor
    duration: Tensor


class Acoustic(nn.Module):
    latent_mean: Tensor
    latent_std: Tensor

    def __init__(self, cfg: AcousticConfig) -> None:
        super().__init__()
        d = cfg.d
        self.cfg, self.dh = cfg, d // cfg.n_heads
        self.text = TextEncoder(cfg)
        self.speaker = nn.Embedding(cfg.n_speakers, d)
        self.quality = nn.Embedding(cfg.n_quality, d)
        self.duration = Duration(d, cfg.dur_hidden)
        self.aligner = Aligner(cfg)
        self.in_proj = nn.Linear(cfg.latent_dim, d)
        self.t_embed = TimestepEmbed(d)
        self.ada_shared = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.blocks = nn.ModuleList(DiTBlock(d, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.n_layers))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out_proj = nn.Linear(d, cfg.latent_dim)
        # Latent normalization, filled from the training set's statistics and saved with the weights.
        self.register_buffer("latent_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("latent_std", torch.ones(cfg.latent_dim))
        self._init()

    def _init(self) -> None:
        for emb in (self.speaker, self.quality):
            nn.init.normal_(emb.weight, std=0.02)
        for zero in (self.ada_shared[1], self.ada_out[1], self.out_proj):  # adaLN-zero: blocks start as identity
            assert isinstance(zero, nn.Linear)
            nn.init.zeros_(zero.weight)
            nn.init.zeros_(zero.bias)

    # ---- stages -------------------------------------------------------------------------------------------------

    def text_stage(self, ids: Int[Tensor, "b l"], mask: Bool[Tensor, "b l"], speaker: Int[Tensor, " b"],
                   quality: Int[Tensor, " b"]) -> TextOut:
        g = self.speaker(speaker) + self.quality(quality)
        h = self.text(ids, mask) + g[:, None]
        return TextOut(h, self.duration(h, mask), g)

    @staticmethod
    def frames_from_log(log_dur: Tensor, mask: Tensor) -> Tensor:
        return torch.expm1(log_dur.clamp(max=MAX_LOG_DUR)).clamp(min=1e-3) * mask.float()

    def condition(self, h: Tensor, dur: Float[Tensor, "b l"], mask: Tensor, cw: Tensor, wstart: Tensor,
                  fw: Tensor, fp: Tensor) -> Float[Tensor, "b t d"]:
        return self.aligner(h, cw, letter_positions(dur, mask, cw, wstart), fw, fp, mask)

    def velocity(self, x: Float[Tensor, "b t c"], cond: Float[Tensor, "b t d"], t: Float[Tensor, " b"],
                 g: Float[Tensor, "b d"], fmask: Bool[Tensor, "b t"]) -> Float[Tensor, "b t c"]:
        cos, sin = rope(torch.arange(x.shape[1], device=x.device), self.dh)
        c = self.t_embed(t) + g
        mod = self.ada_shared(c).view(-1, 6, self.cfg.d)
        y = self.in_proj(x) + cond
        for block in self.blocks:
            y = block(y, mod, cos, sin, fmask)
        s, gate = self.ada_out(c).chunk(2, -1)
        return self.out_proj(self.norm_out(y) * (1 + gate.unsqueeze(1)) + s.unsqueeze(1))

    # ---- training -----------------------------------------------------------------------------------------------

    def forward(self, **batch: Tensor | float) -> Losses:
        """The training loss, so DistributedDataParallel and torch.compile wrap it like any forward pass."""
        return self.loss(**batch)  # type: ignore[arg-type]

    def loss(self, ids: Tensor, mask: Tensor, cw: Tensor, wstart: Tensor, dur: Tensor, fw: Tensor, fp: Tensor,
             fmask: Tensor, latents: Float[Tensor, "b t c"], speaker: Tensor, quality: Tensor,
             cond_drop: float = 0.1, dur_weight: float = 1.0) -> Losses:
        """Flow-matching loss on normalized latents, plus letter-duration regression in log space."""
        text = self.text_stage(ids, mask, speaker, quality)
        m = mask.float()
        dur_loss = ((text.log_dur - torch.log1p(dur)) ** 2 * m).sum() / m.sum()
        cond = self.condition(text.h, dur, mask, cw, wstart, fw, fp)
        b = ids.shape[0]
        keep = (torch.rand(b, device=ids.device) >= cond_drop).float()[:, None, None]  # for classifier-free guidance
        x1 = (latents - self.latent_mean) / self.latent_std
        x0 = torch.randn_like(x1)
        t = torch.sigmoid(torch.randn(b, device=ids.device))  # logit-normal: more steps mid-trajectory
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        v = self.velocity(xt, cond * keep, t, text.g, fmask)
        fm = fmask.float()
        flow = (((v - (x1 - x0)) ** 2).mean(-1) * fm).sum() / fm.sum()
        return Losses(flow + dur_weight * dur_loss, flow, dur_loss)

    # ---- sampling -----------------------------------------------------------------------------------------------

    def student_x1(self, x: Tensor, cond: Tensor, g: Tensor, fmask: Tensor, t: float) -> Tensor:
        """A distilled student's prediction of clean (normalized) latents from x_t in one network pass."""
        tt = torch.full((x.shape[0],), t, device=x.device)
        return x + (1 - t) * self.velocity(x, cond, tt, g, fmask)

    @torch.no_grad()
    def sample_distilled(self, cond: Tensor, g: Tensor, fmask: Tensor, noises: Tensor,
                         times: tuple[float, ...]) -> Float[Tensor, "b t c"]:
        """Few-step sampling of a DMD2 student: predict x1, re-noise to the next time, repeat. noises: (steps, b, t, c).
        Returns de-normalized latents."""
        x = noises[0]
        x1 = x
        for k, t in enumerate(times):
            x1 = self.student_x1(x, cond, g, fmask, t)
            if k + 1 < len(times):
                x = (1 - times[k + 1]) * noises[k + 1] + times[k + 1] * x1
        return x1 * self.latent_std + self.latent_mean

    @torch.no_grad()
    def sample(self, cond: Tensor, g: Tensor, fmask: Tensor, noise: Tensor, steps: int = 16,
               guidance: float = 2.0) -> Float[Tensor, "b t c"]:
        """Midpoint ODE solver with classifier-free guidance on the text condition. Returns de-normalized latents."""
        x = noise
        dt = 1.0 / steps
        both_cond = torch.cat([cond, torch.zeros_like(cond)])
        both_g, both_mask = torch.cat([g, g]), torch.cat([fmask, fmask])

        def f(x: Tensor, t: float) -> Tensor:
            tt = torch.full((2 * x.shape[0],), t, device=x.device)
            vc, vu = self.velocity(torch.cat([x, x]), both_cond, tt, both_g, both_mask).chunk(2)
            return vu + guidance * (vc - vu)

        for k in range(steps):
            t = k * dt
            x = x + dt * f(x + 0.5 * dt * f(x, t), t + 0.5 * dt)
        return x * self.latent_std + self.latent_mean
