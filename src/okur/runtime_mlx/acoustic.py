"""Apple MLX port of the acoustic model, for inference on Apple Silicon. Loads the same weights as the PyTorch model.

Mirrors okur.model.acoustic operation for operation; tests/test_mlx.py checks the two agree. Scatter/gather over
words is done with one-hot matrices, which is simple and fast at sentence sizes.
"""

import math
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

from okur.model.acoustic import MAX_LOG_DUR
from okur.model.config import AcousticConfig

NEG = -1e4


def rope(pos: mx.array, dim: int) -> tuple[mx.array, mx.array]:
    inv = 1.0 / (10000.0 ** (mx.arange(dim // 2).astype(mx.float32) / (dim // 2)))
    ang = pos[..., None].astype(mx.float32) * inv
    ang = mx.concatenate([ang, ang], -1)
    return mx.cos(ang), mx.sin(ang)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    a, b = mx.split(x, 2, axis=-1)
    return x * cos + mx.concatenate([-b, a], -1) * sin


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float = 4.0) -> None:
        super().__init__()
        hidden = int(d * mult * 2 / 3)
        self.w1, self.w3 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask: mx.array) -> mx.array:
        b, t, d = x.shape
        q, k, v = (y.reshape(b, t, self.h, self.dh).transpose(0, 2, 1, 3) for y in mx.split(self.qkv(x), 3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        bias = mx.where(mask[:, None, None, :], 0.0, -mx.inf)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(self.dh), mask=bias)
        return self.proj(y.transpose(0, 2, 1, 3).reshape(b, t, d))


class ConvNeXtBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1, self.pw2 = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)
        self.gamma = mx.ones(d)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        y = self.norm(self.dw(x * mask[..., None]))
        return x + self.gamma * self.pw2(nn.gelu(self.pw1(y)))


class TextBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.ff = Attention(d, heads), SwiGLU(d)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.n1(x), cos, sin, mask)
        return x + self.ff(self.n2(x))


def masked_group_norm(x: mx.array, weight: mx.array, bias: mx.array, m: mx.array, eps: float = 1e-5) -> mx.array:
    """x: (b, l, c); statistics over real positions and all channels."""
    count = m.sum((1, 2), keepdims=True) * x.shape[-1]
    mean = (x * m).sum((1, 2), keepdims=True) / count
    var = (((x - mean) * m) ** 2).sum((1, 2), keepdims=True) / count
    return (x - mean) / mx.sqrt(var + eps) * weight + bias


class Duration(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.c1, self.c2 = nn.Conv1d(d, hidden, 3, padding=1), nn.Conv1d(hidden, hidden, 3, padding=1)
        self.n1_weight, self.n1_bias = mx.ones(hidden), mx.zeros(hidden)
        self.n2_weight, self.n2_bias = mx.ones(hidden), mx.zeros(hidden)
        self.out = nn.Linear(hidden, 1)

    def __call__(self, h: mx.array, mask: mx.array) -> mx.array:
        m = mask[..., None].astype(mx.float32)
        x = masked_group_norm(nn.silu(self.c1(h * m)), self.n1_weight, self.n1_bias, m) * m
        x = masked_group_norm(nn.silu(self.c2(x)), self.n2_weight, self.n2_bias, m)
        return mx.where(mask, self.out(x).squeeze(-1), 0.0)


def _one_hot(index: mx.array, n: int) -> mx.array:
    return (index[..., None] == mx.arange(n)).astype(mx.float32)


class Aligner(nn.Module):
    def __init__(self, cfg: AcousticConfig) -> None:
        super().__init__()
        d = cfg.d
        self.h, self.dh = cfg.align_heads, d // cfg.align_heads
        self.pos_scale, self.lookback, self.lookahead = cfg.pos_scale, cfg.lookback, cfg.lookahead
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.frame_q = mx.zeros((1, 1, d))
        self.log_sigma, self.bias_w, self.log_temp = mx.zeros(self.h), mx.ones(self.h), mx.zeros(())

    def __call__(self, h: mx.array, cw: mx.array, cp: mx.array, fw: mx.array, fp: mx.array,
                 mask: mx.array) -> mx.array:
        b, n_letters, d = h.shape
        t = fw.shape[1]
        c, f = mx.maximum(cw, 0), mx.maximum(fw, 0)
        oh = _one_hot(c, n_letters) * (cw >= 0)[..., None]
        wlen = mx.maximum(oh.sum(1), 1.0)
        woff = mx.cumsum(wlen, -1) - wlen
        cg = mx.take_along_axis(woff, c, 1) + cp * mx.take_along_axis(wlen, c, 1)
        fg = mx.take_along_axis(woff, f, 1) + fp * mx.take_along_axis(wlen, f, 1)
        q = self.q(mx.broadcast_to(self.frame_q, (b, t, d))).reshape(b, t, self.h, self.dh).transpose(0, 2, 1, 3)
        k = self.k(h).reshape(b, n_letters, self.h, self.dh).transpose(0, 2, 1, 3)
        v = self.v(h).reshape(b, n_letters, self.h, self.dh).transpose(0, 2, 1, 3)
        q = apply_rope(q, *(x[:, None] for x in rope(fg * self.pos_scale, self.dh)))
        k = apply_rope(k, *(x[:, None] for x in rope(cg * self.pos_scale, self.dh)))
        logits = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.dh) * mx.exp(self.log_temp)
        sig2 = (mx.exp(self.log_sigma) ** 2).reshape(1, self.h, 1, 1)
        dist2 = ((fg[:, :, None] - cg[:, None, :]) ** 2)[:, None]
        logits = logits - self.bias_w.reshape(1, self.h, 1, 1) * dist2 / (2 * sig2)
        rel = cw[:, None, :] - fw[:, :, None]
        allow = (rel >= -self.lookback) & (rel <= self.lookahead) & mask[:, None, :] & (cw[:, None, :] >= 0)
        attn = mx.softmax(mx.where(allow[:, None], logits, NEG), -1) * allow[:, None]
        return self.o((attn @ v).transpose(0, 2, 1, 3).reshape(b, t, d))


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: float) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d, eps=1e-6, affine=False)
        self.n2 = nn.LayerNorm(d, eps=1e-6, affine=False)
        self.attn, self.ff = Attention(d, heads), SwiGLU(d, ff_mult)
        self.ada_offset = mx.zeros((6, d))

    def __call__(self, x: mx.array, mod: mx.array, cos: mx.array, sin: mx.array, mask: mx.array) -> mx.array:
        sa, ga, aa, sf, gf, af = (y[:, None] for y in mx.split(mod + self.ada_offset[None], 6, axis=1))
        sa, ga, aa, sf, gf, af = (y.squeeze(2) for y in (sa, ga, aa, sf, gf, af))
        x = x + aa * self.attn(self.n1(x) * (1 + ga) + sa, cos, sin, mask)
        return x + af * self.ff(self.n2(x) * (1 + gf) + sf)


class Acoustic(nn.Module):
    def __init__(self, cfg: AcousticConfig) -> None:
        super().__init__()
        d = cfg.d
        self.cfg, self.dh = cfg, d // cfg.n_heads
        self.text_emb = nn.Embedding(cfg.vocab_size, d)
        self.text_conv = [ConvNeXtBlock(d) for _ in range(cfg.text_conv)]
        self.text_attn = [TextBlock(d, cfg.n_heads) for _ in range(cfg.text_attn)]
        self.text_norm = nn.LayerNorm(d)
        self.speaker = nn.Embedding(cfg.n_speakers, d)
        self.quality = nn.Embedding(cfg.n_quality, d)
        self.duration = Duration(d, cfg.dur_hidden)
        self.aligner = Aligner(cfg)
        self.in_proj = nn.Linear(cfg.latent_dim, d)
        self.t_mlp0, self.t_mlp2 = nn.Linear(256, d), nn.Linear(d, d)
        self.ada_shared = nn.Linear(d, 6 * d)
        self.blocks = [DiTBlock(d, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.n_layers)]
        self.norm_out = nn.LayerNorm(d, eps=1e-6, affine=False)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out_proj = nn.Linear(d, cfg.latent_dim)
        self.latent_mean, self.latent_std = mx.zeros(cfg.latent_dim), mx.ones(cfg.latent_dim)

    @classmethod
    def from_torch_state(cls, cfg: AcousticConfig, state: dict[str, Any]) -> "Acoustic":
        model = cls(cfg)
        model.load_weights(list(convert_state(state).items()))
        mx.eval(model.parameters())
        return model

    def text_stage(self, ids: mx.array, mask: mx.array, speaker: mx.array,
                   quality: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        g = self.speaker(speaker) + self.quality(quality)
        x = self.text_emb(ids)
        for block in self.text_conv:
            x = block(x, mask)
        if self.text_attn:
            cos, sin = rope(mx.arange(ids.shape[1]), self.dh)
            for block in self.text_attn:
                x = block(x, cos, sin, mask)
        h = self.text_norm(x) + g[:, None]
        return h, self.duration(h, mask), g

    @staticmethod
    def frames_from_log(log_dur: mx.array, mask: mx.array) -> mx.array:
        return mx.maximum(mx.expm1(mx.minimum(log_dur, MAX_LOG_DUR)), 1e-3) * mask

    def condition(self, h: mx.array, dur: mx.array, mask: mx.array, cw: mx.array, wstart: mx.array, fw: mx.array,
                  fp: mx.array) -> mx.array:
        m = mask.astype(mx.float32)
        c = mx.maximum(dur, 1e-4) * m
        done = mx.cumsum(c, -1)
        before = done - c
        word = mx.maximum(cw, 0)
        oh = _one_hot(word, h.shape[1])
        total = mx.take_along_axis((oh * c[..., None]).sum(1), word, 1)
        cp = mx.clip((done - mx.take_along_axis(before, wstart, 1) - 0.5 * c) / mx.maximum(total, 1e-8), 0, 1) * m
        return self.aligner(h, cw, cp, fw, fp, mask)

    def velocity(self, x: mx.array, cond: mx.array, t: mx.array, g: mx.array, fmask: mx.array) -> mx.array:
        cos, sin = rope(mx.arange(x.shape[1]), self.dh)
        half = 128
        a = t[:, None] * 1000.0 * mx.exp(-math.log(10000) * mx.arange(half) / half)[None]
        c = self.t_mlp2(nn.silu(self.t_mlp0(mx.concatenate([mx.cos(a), mx.sin(a)], -1)))) + g
        mod = self.ada_shared(nn.silu(c)).reshape(-1, 6, self.cfg.d)
        y = self.in_proj(x) + cond
        for block in self.blocks:
            y = block(y, mod, cos, sin, fmask)
        s, gate = mx.split(self.ada_out(nn.silu(c)), 2, -1)
        return self.out_proj(self.norm_out(y) * (1 + gate[:, None]) + s[:, None])

    def sample_distilled(self, cond: mx.array, g: mx.array, fmask: mx.array, noises: mx.array,
                         times: tuple[float, ...]) -> mx.array:
        x = noises[0]
        x1 = x
        for k, t in enumerate(times):
            x1 = x + (1 - t) * self.velocity(x, cond, mx.full((x.shape[0],), t), g, fmask)
            if k + 1 < len(times):
                x = (1 - times[k + 1]) * noises[k + 1] + times[k + 1] * x1
        return x1 * self.latent_std + self.latent_mean

    def sample(self, cond: mx.array, g: mx.array, fmask: mx.array, noise: mx.array, steps: int = 16,
               guidance: float = 2.0) -> mx.array:
        x, dt = noise, 1.0 / steps
        both_cond = mx.concatenate([cond, mx.zeros_like(cond)])
        both_g, both_mask = mx.concatenate([g, g]), mx.concatenate([fmask, fmask])

        def f(x: mx.array, t: float) -> mx.array:
            vc, vu = mx.split(self.velocity(mx.concatenate([x, x]), both_cond, mx.full((2 * x.shape[0],), t), both_g,
                                            both_mask), 2)
            return vu + guidance * (vc - vu)

        for k in range(steps):
            t = k * dt
            x = x + dt * f(x + 0.5 * dt * f(x, t), t + 0.5 * dt)
        return x * self.latent_std + self.latent_mean


def convert_state(state: dict[str, Any]) -> dict[str, mx.array]:
    """PyTorch state dict → MLX parameter names and layouts (conv weights are (out, k, in) in MLX)."""
    out: dict[str, mx.array] = {}
    for key, value in state.items():
        a = np.asarray(value.detach().cpu().float().numpy() if hasattr(value, "detach") else value)
        name = (key.replace("text.emb.", "text_emb.").replace("text.conv.", "text_conv.")
                .replace("text.attn.", "text_attn.").replace("text.norm.", "text_norm.")
                .replace("t_embed.mlp.0.", "t_mlp0.").replace("t_embed.mlp.2.", "t_mlp2.")
                .replace("ada_shared.1.", "ada_shared.").replace("ada_out.1.", "ada_out.")
                .replace("duration.n1.weight", "duration.n1_weight").replace("duration.n1.bias", "duration.n1_bias")
                .replace("duration.n2.weight", "duration.n2_weight").replace("duration.n2.bias", "duration.n2_bias"))
        if a.ndim == 3 and name.endswith(".weight"):  # Conv1d (out, in, k) → (out, k, in)
            a = a.transpose(0, 2, 1)
        out[name] = mx.array(a)
    return out
