"""The MLX port computes what the PyTorch model computes, on the same weights."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core", reason="MLX runs on Apple silicon only", exc_type=ImportError)

from okur.frontend import encode
from okur.model.acoustic import Acoustic as TorchAcoustic
from okur.model.config import AcousticConfig
from okur.model.timeline import frame_timeline, word_frames, words
from okur.runtime_mlx.acoustic import Acoustic as MlxAcoustic

TOL = dict(atol=2e-4, rtol=2e-3)


@pytest.fixture(scope="module")
def models() -> tuple[TorchAcoustic, MlxAcoustic]:
    torch.manual_seed(0)
    cfg = AcousticConfig(n_speakers=2)
    tm = TorchAcoustic(cfg).eval()
    for p in tm.parameters():
        p.data.add_(torch.randn_like(p) * 0.02)
    tm.latent_std.fill_(1.3)
    return tm, MlxAcoustic.from_torch_state(cfg, tm.state_dict())


def test_parity(models: tuple[TorchAcoustic, MlxAcoustic]) -> None:
    tm, mm = models
    letters = "bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor."
    ids = torch.tensor([encode(letters)])
    mask = torch.ones_like(ids, dtype=torch.bool)
    spk, q = torch.tensor([1]), torch.tensor([0])
    w = words(letters)
    with torch.no_grad():
        h, log_dur, g = tm.text_stage(ids, mask, spk, q)
        dur = tm.frames_from_log(log_dur, mask)
        fw, fp = frame_timeline(word_frames(dur[0], w.cw, w.n_words))
        cond = tm.condition(h, dur, mask, w.cw[None], w.wstart[None], fw[None], fp[None])
        noise = torch.randn(1, len(fw), 64, generator=torch.Generator().manual_seed(0))
        fmask = torch.ones(1, len(fw), dtype=torch.bool)
        z = tm.sample(cond, g, fmask, noise, steps=4)

    def m(x: torch.Tensor) -> mx.array:
        return mx.array(x.numpy())

    mh, mlog, mg = mm.text_stage(m(ids), m(mask), m(spk), m(q))
    np.testing.assert_allclose(np.array(mlog), log_dur.numpy(), **TOL)
    mcond = mm.condition(mh, m(dur), m(mask), m(w.cw[None]), m(w.wstart[None]), m(fw[None]), m(fp[None]))
    np.testing.assert_allclose(np.array(mcond), cond.numpy(), **TOL)
    mz = mm.sample(mcond, mg, m(fmask), m(noise), steps=4)
    np.testing.assert_allclose(np.array(mz), z.numpy(), atol=2e-3, rtol=2e-3)


def test_decoder_parity() -> None:
    from okur.model.config import DecoderConfig
    from okur.model.decoder import Decoder as TorchDecoder
    from okur.runtime_mlx.decoder import Decoder as MlxDecoder

    torch.manual_seed(0)
    cfg = DecoderConfig()
    td = TorchDecoder(cfg).eval()
    md = MlxDecoder.from_torch_state(cfg, td.state_dict())
    z = torch.randn(1, 64, 37)
    with torch.no_grad():
        ref = td(z).numpy()
    out = np.array(md(mx.array(z.transpose(1, 2).numpy())))
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=2e-4, rtol=1e-3)
