"""Export the acoustic model and decoder to ONNX for CPU, Android and server runtimes.

Four graphs, each with dynamic batch / letter / frame axes; the sampling loop and the word–letter timeline stay in host
code (a few lines in any language):
  text.onnx       ids, mask, speaker, quality      → h, log_dur, g
  condition.onnx  h, dur, mask, cw, wstart, fw, fp → cond
  velocity.onnx   x, cond, t, g, fmask             → v
  decoder.onnx    z (b, 64, t)                     → audio
"""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from okur.model.acoustic import Acoustic
from okur.model.decoder import Decoder

OPSET = 18


class _Text(nn.Module):
    def __init__(self, m: Acoustic) -> None:
        super().__init__()
        self.m = m

    def forward(self, ids: Tensor, mask: Tensor, speaker: Tensor, quality: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.m.text_stage(ids, mask, speaker, quality)
        return out.h, out.log_dur, out.g


class _Condition(nn.Module):
    def __init__(self, m: Acoustic) -> None:
        super().__init__()
        self.m = m

    def forward(self, h: Tensor, dur: Tensor, mask: Tensor, cw: Tensor, wstart: Tensor, fw: Tensor,
                fp: Tensor) -> Tensor:
        return self.m.condition(h, dur, mask, cw, wstart, fw, fp)


class _Velocity(nn.Module):
    def __init__(self, m: Acoustic) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: Tensor, cond: Tensor, t: Tensor, g: Tensor, fmask: Tensor) -> Tensor:
        return self.m.velocity(x, cond, t, g, fmask)


def export(acoustic: Acoustic, decoder: Decoder | None, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    acoustic = acoustic.eval().cpu()
    b, n_letters, t, d = 2, 12, 30, acoustic.cfg.d
    ids = torch.randint(2, 40, (b, n_letters))
    mask = torch.ones(b, n_letters, dtype=torch.bool)
    spk = torch.zeros(b, dtype=torch.long)
    cw = torch.arange(n_letters).div(4, rounding_mode="floor").repeat(b, 1)
    fw = torch.arange(t).div(10, rounding_mode="floor").repeat(b, 1)
    fp = (torch.arange(t) % 10 / 10).float().repeat(b, 1)
    fmask = torch.ones(b, t, dtype=torch.bool)
    batch = torch.export.Dim("batch", min=1, max=256)
    letters, frames = torch.export.Dim("letters", min=2, max=4096), torch.export.Dim("frames", min=2, max=6000)
    graphs: list[tuple[str, nn.Module, tuple[Tensor, ...], list[str], list[str], tuple[dict[int, Any], ...]]] = [
        # speaker and quality must be distinct tensors: the exporter merges inputs that are the same object.
        ("text", _Text(acoustic), (ids, mask, spk, spk.clone()), ["ids", "mask", "speaker", "quality"],
         ["h", "log_dur", "g"],
         ({0: batch, 1: letters}, {0: batch, 1: letters}, {0: batch}, {0: batch})),
        ("condition", _Condition(acoustic),
         (torch.randn(b, n_letters, d), torch.rand(b, n_letters) + 1, mask, cw, cw * 4, fw, fp),
         ["h", "dur", "mask", "cw", "wstart", "fw", "fp"], ["cond"],
         (*({0: batch, 1: letters} for _ in range(5)), {0: batch, 1: frames}, {0: batch, 1: frames})),
        ("velocity", _Velocity(acoustic), (torch.randn(b, t, 64), torch.randn(b, t, d), torch.rand(b),
                                           torch.randn(b, d), fmask),
         ["x", "cond", "t", "g", "fmask"], ["v"],
         ({0: batch, 1: frames}, {0: batch, 1: frames}, {0: batch}, {0: batch}, {0: batch, 1: frames})),
    ]
    if decoder is not None:
        graphs.append(("decoder", decoder.eval().cpu(), (torch.randn(b, 64, t),), ["z"], ["audio"],
                       ({0: batch, 2: frames},)))
    paths = []
    for name, module, args, inputs, outputs, shapes in graphs:
        path = out_dir / f"{name}.onnx"
        torch.onnx.export(module, args, path, input_names=inputs, output_names=outputs, dynamic_shapes=shapes,
                          opset_version=OPSET, dynamo=True, external_data=False)
        paths.append(path)
    return paths
