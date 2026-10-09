import logging
from pathlib import Path
from typing import Literal

from cyclopts import App

app = App(name="okur", help="Turkish TTS: data preparation, training and synthesis.")


@app.command
def prepare(source: str, raw_root: Path, out_dir: Path, *, shard_size: int = 128, device: str = "mps",
            workers: int = 8, rank: int = 0, world: int = 1, limit: int | None = None, align_batch: int = 8,
            encode_seconds: float = 15.0, tf32: bool = True, recognizer_bf16: bool = True) -> None:
    """Turn a raw corpus into training shards (resumable; one process per GPU with --rank/--world)."""
    from okur.data.prepare import prepare as run

    run(source, raw_root, out_dir, shard_size=shard_size, device=device, workers=workers, rank=rank, world=world,
        limit=limit, align_batch=align_batch, encode_seconds=encode_seconds, tf32=tf32, recognizer_bf16=recognizer_bf16)


@app.command
def train(config: Path) -> None:
    """Train the acoustic model from a YAML config. Resumes automatically from the run directory's last checkpoint."""
    from okur.train.trainer import main

    main(config)


@app.command
def train_decoder(config: Path) -> None:
    """Train the latent-to-audio decoder (GAN) from a YAML config. Resumes automatically."""
    from okur.train.decoder_trainer import main

    main(config)


@app.command
def distill(config: Path) -> None:
    """DMD2: distill a 4-step student from a teacher checkpoint (YAML config). Resumes automatically."""
    from okur.train.distill import main

    main(config)


@app.command
def bench(data: list[Path], *, out: Path = Path("runs/bench"), device: str = "cuda", max_frames: int = 40000,
          decoder_batch: int = 32, warmup: int = 20, steps: int = 50, clips: int = 64) -> None:
    """Measure every speed option (training, decoder GAN, data prep) on this machine. Run first on a new GPU."""
    from okur.bench import run

    run(data, out, device=device, max_frames=max_frames, decoder_batch=decoder_batch, warmup=warmup, steps=steps,
        clips=clips)


@app.command
def eval_freya(checkpoint: Path, shards: list[Path], *, out: Path = Path("runs/eval_freya"),
               decoder: Path | None = None, device: str = "cuda", steps: int = 16, guidance: float = 3.0,
               ema: bool = True, speaker: str | None = None, whisper_batch: int = 16) -> None:
    """Freya-TR-Eval (495 sentences, 3 seeds, Whisper-large-v3): ours vs EMA Lightning, with contamination check."""
    from okur.evals.freya import run

    run(checkpoint, shards, out, decoder=decoder, device=device, steps=steps, guidance=guidance, with_ema=ema,
        speaker=speaker, whisper_batch=whisper_batch)


@app.command
def say(text: str, *, model: Path = Path("release"), out: Path = Path("speech.wav"), speed: float = 1.0, seed: int = 0,
        runtime: Literal["onnx", "torch", "mlx"] = "onnx") -> None:
    """Speak Turkish text into a 48 kHz WAV file with a release model (see `okur release`)."""
    import time
    import wave

    import numpy as np

    started = time.perf_counter()
    if runtime == "onnx":
        from okur.export_web import OnnxSynthesizer

        audio = OnnxSynthesizer(model / "onnx").say(text, speed=speed, seed=seed)
    elif runtime == "mlx":
        from okur.runtime_mlx.synth import MlxSynthesizer

        audio = MlxSynthesizer(model / "acoustic.safetensors", model / "decoder.safetensors").say(
            text, speed=speed, seed=seed)
    else:
        from okur.synth import torch_synthesizer

        audio = torch_synthesizer(model / "acoustic.safetensors", model / "decoder.safetensors")(
            text, speed=speed, seed=seed)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm.tobytes())
    print(f"{out}: {len(pcm) / 48000:.1f} s of audio in {time.perf_counter() - started:.1f} s")


@app.command
def release(acoustic: Path, decoder: Path, voice: str, out: Path = Path("release"), *, onnx: bool = True) -> None:
    """Package one voice of a distilled model for publishing: safetensors + JSON, and ONNX graphs in <out>/onnx."""
    from okur.release import write

    paths = write(acoustic, decoder, voice, out)
    if onnx:
        from okur.export_web import export

        export(out / "acoustic.safetensors", out / "decoder.safetensors", voice, out / "onnx")
    for p in paths:
        print(p)


@app.command
def export_web(acoustic: Path, decoder: Path, voice: str, out: Path = Path("runs/web_export/fp16"), *,
               fp16: bool = True) -> None:
    """Export graphs for ONNX Runtime Web / Mobile (batch 1, one voice, fp16-stored weights)."""
    from okur.export_web import export

    print(export(acoustic, decoder, voice, out, fp16=fp16))


def _setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "transformers", "diffusers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    _setup()
    app()
