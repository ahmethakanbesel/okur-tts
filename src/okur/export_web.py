"""Export one voice of a distilled model for browsers and phones (ONNX Runtime Web / Mobile), plus a reference
implementation of the host loop that the TypeScript runtime mirrors.

Three graphs, batch 1, voice and sampling schedule baked in, weights stored as fp16 (cast to fp32 when the session
loads, so compute stays fp32 and every file stays far below Cloudflare Pages' 25 MiB limit):
  text.onnx    ids (1, L) int64                                            → h (1, L, d), dur (1, L) frames
  sample.onnx  h, dur, cw, wstart (1, L), fw, fp (1, T), noise (S, 1, T, C) → z (1, C, T) de-normalized latents
  decoder.onnx z (1, C, T)                                                 → audio (1, T · hop)
The host does what is cheap and data-dependent: the text frontend, speed, the word timeline and the noise.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from okur import release
from okur.frontend import SYMBOLS, Frontend, encode
from okur.model.acoustic import Acoustic
from okur.model.config import AcousticConfig, DecoderConfig
from okur.model.decoder import Decoder
from okur.model.timeline import frame_timeline, word_frames, words

OPSET = 18
FP16_MIN_ELEMENTS = 1024  # small tensors (norm scales, biases) stay fp32


class _Text(nn.Module):
    g: Tensor

    def __init__(self, m: Acoustic, g: Tensor) -> None:
        super().__init__()
        self.m = m
        self.register_buffer("g", g)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        mask = torch.ones_like(ids, dtype=torch.bool)
        h = self.m.text(ids, mask) + self.g[:, None]
        return h, self.m.frames_from_log(self.m.duration(h, mask), mask)


class _Sample(nn.Module):
    g: Tensor

    def __init__(self, m: Acoustic, g: Tensor, times: tuple[float, ...]) -> None:
        super().__init__()
        self.m, self.times = m, times
        self.register_buffer("g", g)

    def forward(self, h: Tensor, dur: Tensor, cw: Tensor, wstart: Tensor, fw: Tensor, fp: Tensor,
                noise: Tensor) -> Tensor:
        mask = torch.ones_like(dur, dtype=torch.bool)
        fmask = torch.ones_like(fp, dtype=torch.bool)
        cond = self.m.condition(h, dur, mask, cw, wstart, fw, fp)
        x = noise[0]
        x1 = x
        for k, t in enumerate(self.times):  # Acoustic.sample_distilled, unrolled
            x1 = x + (1 - t) * self.m.velocity(x, cond, torch.full((1,), t), self.g, fmask)
            if k + 1 < len(self.times):
                x = (1 - self.times[k + 1]) * noise[k + 1] + self.times[k + 1] * x1
        return (x1 * self.m.latent_std + self.m.latent_mean).transpose(1, 2)


class PolyphaseUp(nn.Module):
    """ConvTranspose1d rewritten as one ordinary Conv1d that computes all `stride` output phases, then an interleave.

    Same numbers (tests/test_release.py checks it), but ONNX Runtime's CPU and WebAssembly backends run Conv much
    faster than ConvTranspose. Output phase j of y[n·s + j] uses taps k ≡ j + p (mod s) at input offsets
    n + ⌊(j+p)/s⌋ − m.
    """

    def __init__(self, up: nn.ConvTranspose1d) -> None:
        super().__init__()
        w = up.weight.detach()  # (in, out, k)
        c_in, c_out, k = w.shape
        s, p = up.stride[0], int(up.padding[0])
        taps: list[tuple[int, int, int]] = []  # (phase, kernel tap, input offset)
        for j in range(s):
            q = (j + p) % s
            c = (j + p) // s
            taps += [(j, q + m * s, c - m) for m in range((k - q + s - 1) // s)]
        lo, hi = min(t[2] for t in taps), max(t[2] for t in taps)
        weight = torch.zeros(s * c_out, c_in, hi - lo + 1)
        for j, tap, offset in taps:
            weight[j * c_out:(j + 1) * c_out, :, offset - lo] = w[:, :, tap].T
        bias = up.bias.detach() if up.bias is not None else torch.zeros(c_out)
        self.conv = nn.Conv1d(c_in, s * c_out, hi - lo + 1)
        self.conv.weight = nn.Parameter(weight)
        self.conv.bias = nn.Parameter(bias.repeat(s))
        # Output length is n·s + tail; computing n + extra phases needs `extra` more zero frames on the right.
        self.stride, self.c_out, self.tail = s, c_out, k - 2 * p - s
        self.extra = max(0, -(-self.tail // s))
        self.pad = (max(0, -lo), max(0, hi + self.extra))
        self.shift = lo + self.pad[0]  # conv output row holding phase index 0

    def forward(self, x: Tensor) -> Tensor:
        b, _, n = x.shape
        phases = n + self.extra
        y = self.conv(F.pad(x, self.pad))[..., self.shift:self.shift + phases]
        y = y.reshape(b, self.stride, self.c_out, phases).permute(0, 2, 3, 1)
        return y.reshape(b, self.c_out, phases * self.stride)[..., :n * self.stride + self.tail]


class _Decoder(nn.Module):
    """okur.model.decoder.Decoder.forward without lengths, upsampling with PolyphaseUp."""

    def __init__(self, dec: Decoder) -> None:
        super().__init__()
        self.dec = dec
        self.ups = nn.ModuleList(PolyphaseUp(up) for up in dec.ups)  # type: ignore[arg-type]

    def forward(self, z: Tensor) -> Tensor:
        d = self.dec
        frames = z.shape[-1]
        x = d.pre(z)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, 0.1))
            blocks = d.blocks[i * d.nk:(i + 1) * d.nk]
            x = sum((b(x) for b in blocks), torch.zeros_like(x)) / d.nk
        return torch.tanh(d.post(F.leaky_relu(x)))[:, 0, :frames * d.hop]


def load(acoustic: Path, decoder: Path, speaker: str, quality: int = 0) -> tuple[Acoustic, Decoder, Tensor]:
    state = release.read(acoustic)
    cfg = AcousticConfig.model_validate(state["model_config"])
    if not cfg.distilled_times:
        raise ValueError("web export needs a distilled (few-step) student")
    model = Acoustic(cfg)
    model.load_state_dict(state["average"])
    dstate = release.read(decoder)
    dec = Decoder(DecoderConfig.model_validate(dstate["model_config"]))
    dec.load_state_dict(dstate["model"])
    model.eval()
    dec.eval()
    spk = torch.tensor([state["speakers"].index(speaker)])
    with torch.no_grad():
        g = model.speaker(spk) + model.quality(torch.tensor([quality]))
    return model, dec, g


def _fp16_weights(path: Path) -> None:
    """Store large fp32 initializers as fp16, each followed by a Cast back to fp32 (folded when a session loads)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(path))
    graph = m.graph
    kept, casts = [], []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and arr.size >= FP16_MIN_ELEMENTS:
            half = numpy_helper.from_array(arr.astype(np.float16), f"{init.name}__fp16")
            kept.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
        else:
            kept.append(init)
    graph.ClearField("initializer")
    graph.initializer.extend(kept)
    nodes = casts + list(graph.node)
    graph.ClearField("node")
    graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def export(acoustic: Path, decoder: Path, speaker: str, out_dir: Path, *, fp16: bool = True) -> dict[str, Any]:
    """Write text/sample/decoder graphs (content-hashed names) and config.json to out_dir; return the config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model, dec, g = load(acoustic, decoder, speaker)
    cfg = model.cfg
    times = tuple(cfg.distilled_times or ())
    n, t = 12, 40
    ids = torch.randint(2, 40, (1, n))
    letters, frames = torch.export.Dim("letters", min=2, max=4096), torch.export.Dim("frames", min=2, max=6000)
    cw = torch.arange(n).div(4, rounding_mode="floor")[None]
    fw = torch.arange(t).div(14, rounding_mode="floor")[None]
    graphs = [
        ("text", _Text(model, g), (ids,), ["ids"], ["h", "dur"], ({1: letters},)),
        ("sample", _Sample(model, g, times),
         (torch.randn(1, n, cfg.d), torch.rand(1, n) + 1, cw, cw * 4, fw, (torch.arange(t) % 14 / 14).float()[None],
          torch.randn(len(times), 1, t, cfg.latent_dim)),
         ["h", "dur", "cw", "wstart", "fw", "fp", "noise"], ["z"],
         ({1: letters}, {1: letters}, {1: letters}, {1: letters}, {1: frames}, {1: frames}, {2: frames})),
        ("decoder", _Decoder(dec), (torch.randn(1, cfg.latent_dim, t),), ["z"], ["audio"], ({2: frames},)),
    ]
    files: dict[str, str] = {}
    for name, module, args, inputs, outputs, shapes in graphs:
        tmp = out_dir / f"{name}.onnx"
        with torch.no_grad():
            torch.onnx.export(module, args, tmp, input_names=inputs, output_names=outputs, dynamic_shapes=shapes,
                              opset_version=OPSET, dynamo=True, external_data=False)
        if fp16:
            _fp16_weights(tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:12]
        final = out_dir / f"{name}.{digest}.onnx"
        for old in out_dir.glob(f"{name}.*.onnx"):
            old.unlink()
        tmp.rename(final)
        files[name] = final.name
    config = {
        "voice": speaker, "sample_rate": 48000, "hop": math.prod(dec.cfg.rates), "latent_dim": cfg.latent_dim,
        "times": list(times), "symbols": list(SYMBOLS), "files": files,
        "bytes": {k: (out_dir / v).stat().st_size for k, v in files.items()},
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    return config


# ---- the host loop, as the TypeScript runtime implements it ------------------------------------------------------

def gaussian_noise(seed: int, count: int) -> np.ndarray:
    """Seeded standard normals that the TypeScript runtime reproduces bit for bit: mulberry32 + Box–Muller."""
    state = seed & 0xFFFFFFFF
    out = np.empty(count, dtype=np.float32)

    def uniform() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        z = state
        z = ((z ^ (z >> 15)) * (z | 1)) & 0xFFFFFFFF
        z ^= (z + (((z ^ (z >> 7)) * (z | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((z ^ (z >> 14)) & 0xFFFFFFFF) / 4294967296.0

    for i in range(0, count, 2):
        u1, u2 = 1.0 - uniform(), uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        out[i] = r * math.cos(2 * math.pi * u2)
        if i + 1 < count:
            out[i + 1] = r * math.sin(2 * math.pi * u2)
    return out


def timeline(letters: str, dur: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = words(letters)
    fw, fp = frame_timeline(word_frames(torch.from_numpy(dur.astype(np.float32)), w.cw, w.n_words))
    return w.cw.numpy(), w.wstart.numpy(), fw.numpy(), fp.numpy()


class OnnxSynthesizer:
    """Runs the exported graphs with onnxruntime exactly as the browser does (reference for tests)."""

    def __init__(self, out_dir: Path, threads: int = 0) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0: ONNX Runtime's default (all cores)
        self.config = json.loads((out_dir / "config.json").read_text())
        self.sessions = {k: ort.InferenceSession(str(out_dir / v), options, providers=["CPUExecutionProvider"])
                         for k, v in self.config["files"].items() if v.endswith(".onnx")}
        self.frontend = Frontend()

    def say(self, text: str, *, speed: float = 1.0, seed: int = 0) -> np.ndarray:
        letters = self.frontend(text)
        ids = np.array([encode(letters)], dtype=np.int64)
        h, dur = (np.asarray(x) for x in self.sessions["text"].run(None, {"ids": ids}))
        dur = dur / speed
        cw, wstart, fw, fp = timeline(letters, dur[0])
        steps, c = len(self.config["times"]), self.config["latent_dim"]
        noise = gaussian_noise(seed, steps * len(fw) * c).reshape(steps, 1, len(fw), c)
        (z,) = self.sessions["sample"].run(None, {
            "h": h, "dur": dur.astype(np.float32), "cw": cw[None], "wstart": wstart[None], "fw": fw[None],
            "fp": fp[None].astype(np.float32), "noise": noise})
        (audio,) = self.sessions["decoder"].run(None, {"z": z})
        return np.asarray(audio)[0]
