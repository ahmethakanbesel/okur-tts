from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from okur.frontend.alphabet import SYMBOLS


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AcousticConfig(_Frozen):
    vocab_size: int = len(SYMBOLS)
    d: Annotated[int, Field(gt=0)] = 224
    n_layers: Annotated[int, Field(gt=0)] = 6
    n_heads: Annotated[int, Field(gt=0)] = 8
    latent_dim: int = 64
    text_conv: int = 4
    text_attn: int = 2  # EMA ships 0; ablated
    align_heads: int = 4
    pos_scale: float = 4.0
    lookback: int = 1
    lookahead: int = 1
    dur_hidden: int = 256
    ff_mult: float = 4.0
    n_speakers: Annotated[int, Field(gt=0)] = 1
    n_quality: Annotated[int, Field(gt=0)] = 4  # recording-quality buckets; 0 = clean full-band
    distilled_times: tuple[float, ...] | None = None  # set on a DMD2 student: sample in these few steps, no guidance


class DecoderConfig(_Frozen):
    latent_dim: int = 64
    ch: int = 256
    rates: tuple[int, ...] = (8, 6, 5, 2, 2, 2)
    kernels: tuple[int, ...] = (16, 12, 10, 4, 4, 4)
    rb_kernels: tuple[int, ...] = (3, 5, 9)
    rb_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    sample_rate: int = 48000
