"""A release (one voice, safetensors) speaks exactly like the checkpoint it came from, in every runtime."""

from pathlib import Path

import numpy as np
import pytest
import torch

from okur import release
from okur.export_web import OnnxSynthesizer, PolyphaseUp, _Decoder, export
from okur.model.acoustic import Acoustic
from okur.model.config import AcousticConfig, DecoderConfig
from okur.model.decoder import Decoder
from okur.synth import torch_synthesizer

TEXT = "Bu konuyla alakalı olarak şirketin karı hala artıyor."


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    torch.manual_seed(0)
    acoustic = Acoustic(AcousticConfig(n_speakers=3, distilled_times=(0.0, 0.25, 0.5, 0.75))).eval()
    for p in acoustic.parameters():
        p.data.add_(torch.randn_like(p) * 0.02)
    decoder = Decoder(DecoderConfig()).eval()
    out = tmp_path_factory.mktemp("ckpt")
    torch.save({"model_config": acoustic.cfg.model_dump(), "average": acoustic.state_dict(),
                "speakers": ["a", "b", "c"]}, out / "acoustic.pt")
    torch.save({"model_config": decoder.cfg.model_dump(), "model": decoder.state_dict(), "disc": {}},
               out / "decoder.pt")
    return out / "acoustic.pt", out / "decoder.pt"


def test_release_keeps_one_voice_exactly(checkpoints: tuple[Path, Path], tmp_path: Path) -> None:
    acoustic, decoder = checkpoints
    release.write(acoustic, decoder, "b", tmp_path)
    state = release.read(tmp_path / "acoustic.safetensors")
    assert state["speakers"] == ["b"] and state["model_config"]["n_speakers"] == 1
    original = release.read(acoustic)["average"]["speaker.weight"]
    assert torch.equal(state["average"]["speaker.weight"][0], original[1])
    assert "disc" not in release.read(tmp_path / "decoder.safetensors")

    full = torch_synthesizer(acoustic, decoder, "cpu")
    single = torch_synthesizer(tmp_path / "acoustic.safetensors", tmp_path / "decoder.safetensors", "cpu")
    np.testing.assert_array_equal(full(TEXT, speaker=1, seed=3), single(TEXT, speaker=0, seed=3))


def test_onnx_web_graphs_match_pytorch(checkpoints: tuple[Path, Path], tmp_path: Path) -> None:
    acoustic, decoder = checkpoints
    release.write(acoustic, decoder, "c", tmp_path)
    export(tmp_path / "acoustic.safetensors", tmp_path / "decoder.safetensors", "c", tmp_path / "onnx", fp16=False)
    onnx_audio = OnnxSynthesizer(tmp_path / "onnx").say(TEXT, seed=5)
    assert onnx_audio.shape[0] > 48000 // 2 and np.isfinite(onnx_audio).all()
    with pytest.raises(ValueError, match="unknown format"):
        (tmp_path / "bad.json").write_text('{"format": "other"}')
        (tmp_path / "bad.safetensors").write_bytes((tmp_path / "decoder.safetensors").read_bytes())
        release.read(tmp_path / "bad.safetensors")


@pytest.mark.parametrize(("stride", "kernel"), [(8, 16), (6, 12), (5, 10), (2, 4), (3, 7), (4, 5)])
def test_polyphase_upsampling_equals_transposed_conv(stride: int, kernel: int) -> None:
    torch.manual_seed(stride * 100 + kernel)
    up = torch.nn.ConvTranspose1d(6, 4, kernel, stride, padding=(kernel - stride) // 2)
    x = torch.randn(2, 6, 13)
    with torch.no_grad():
        torch.testing.assert_close(PolyphaseUp(up)(x), up(x), atol=1e-5, rtol=1e-5)


def test_polyphase_decoder_equals_decoder() -> None:
    torch.manual_seed(0)
    decoder = Decoder(DecoderConfig()).eval()
    z = torch.randn(1, 64, 17)
    with torch.no_grad():
        torch.testing.assert_close(_Decoder(decoder)(z), decoder(z), atol=1e-5, rtol=1e-4)
