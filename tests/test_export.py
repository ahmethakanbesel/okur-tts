"""ONNX graphs compute what the PyTorch modules compute, at shapes other than the ones they were traced with."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from okur.export import export
from okur.frontend import encode
from okur.model.acoustic import Acoustic
from okur.model.config import AcousticConfig, DecoderConfig
from okur.model.decoder import Decoder
from okur.model.timeline import frame_timeline, word_frames, words


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Acoustic, Decoder, Path]:
    torch.manual_seed(0)
    acoustic = Acoustic(AcousticConfig(n_speakers=2)).eval()
    for p in acoustic.parameters():
        p.data.add_(torch.randn_like(p) * 0.02)
    decoder = Decoder(DecoderConfig()).eval()
    out = tmp_path_factory.mktemp("onnx")
    export(acoustic, decoder, out)
    return acoustic, decoder, out


def test_onnx_parity(exported: tuple[Acoustic, Decoder, Path]) -> None:
    acoustic, decoder, out = exported
    letters = "bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor."
    w = words(letters)
    ids = torch.tensor([encode(letters)])
    mask = torch.ones_like(ids, dtype=torch.bool)
    spk = torch.tensor([1])
    run = {n: ort.InferenceSession(str(out / f"{n}.onnx")) for n in ("text", "condition", "velocity", "decoder")}

    with torch.no_grad():
        text = acoustic.text_stage(ids, mask, spk, spk * 0)
        _, log_dur, g = run["text"].run(None, {"ids": ids.numpy(), "mask": mask.numpy(), "speaker": spk.numpy(),
                                                "quality": (spk * 0).numpy()})
        np.testing.assert_allclose(log_dur, text.log_dur.numpy(), atol=1e-4, rtol=1e-3)
        dur = acoustic.frames_from_log(text.log_dur, mask)
        fw, fp = frame_timeline(word_frames(dur[0], w.cw, w.n_words))
        args = (text.h, dur, mask, w.cw[None], w.wstart[None], fw[None], fp[None])
        cond = acoustic.condition(*args)
        (cond_onnx,) = run["condition"].run(None, dict(zip(["h", "dur", "mask", "cw", "wstart", "fw", "fp"],
                                                           [a.numpy() for a in args], strict=True)))
        np.testing.assert_allclose(cond_onnx, cond.numpy(), atol=1e-4, rtol=1e-3)
        x = torch.randn(1, len(fw), 64)
        fmask = torch.ones(1, len(fw), dtype=torch.bool)
        v = acoustic.velocity(x, cond, torch.tensor([0.4]), text.g, fmask)
        (v_onnx,) = run["velocity"].run(None, {"x": x.numpy(), "cond": cond.numpy(), "t": np.array([0.4], np.float32),
                                               "g": g, "fmask": fmask.numpy()})
        np.testing.assert_allclose(v_onnx, v.numpy(), atol=1e-4, rtol=1e-3)
        z = torch.randn(1, 64, 57)
        (audio,) = run["decoder"].run(None, {"z": z.numpy()})
        np.testing.assert_allclose(audio, decoder(z).numpy(), atol=1e-4, rtol=1e-3)
