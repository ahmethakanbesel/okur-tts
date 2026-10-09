"""Building blocks shared by the text encoder and the latent generator."""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn


def rope(pos: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    inv = 1.0 / (10000.0 ** (torch.arange(dim // 2, device=pos.device).float() / (dim // 2)))
    ang = pos[..., None].float() * inv
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat([-b, a], -1) * sin


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float = 4.0) -> None:
        super().__init__()
        hidden = int(d * mult * 2 / 3)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: Float[Tensor, "b t d"], cos: Tensor, sin: Tensor,
                mask: Bool[Tensor, "b t"]) -> Float[Tensor, "b t d"]:
        b, t, d = x.shape
        q, k, v = (y.view(b, t, self.h, self.dh).transpose(1, 2) for y in self.qkv(x).chunk(3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return self.proj(x.transpose(1, 2).reshape(b, t, d))


class ConvNeXtBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.pw2 = nn.Linear(2 * d, d)
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x: Float[Tensor, "b l d"], mask: Bool[Tensor, "b l"]) -> Float[Tensor, "b l d"]:
        y = self.norm(self.dw((x * mask[..., None]).transpose(1, 2)).transpose(1, 2))  # padding reads as zeros
        return x + self.gamma * self.pw2(F.gelu(self.pw1(y)))


class TextBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.ff = Attention(d, heads), SwiGLU(d)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.n1(x), cos, sin, mask)
        return x + self.ff(self.n2(x))


def masked_group_norm(x: Tensor, norm: nn.GroupNorm, m: Tensor) -> Tensor:
    """GroupNorm(1, C) with statistics over real positions only."""
    count = m.sum((1, 2), keepdim=True) * x.shape[1]
    mean = (x * m).sum((1, 2), keepdim=True) / count
    var = (((x - mean) * m) ** 2).sum((1, 2), keepdim=True) / count
    return (x - mean) / torch.sqrt(var + norm.eps) * norm.weight[None, :, None] + norm.bias[None, :, None]
