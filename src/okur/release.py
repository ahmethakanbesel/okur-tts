"""The published model format: safetensors weights and a JSON config per model, one voice, no pickles.

    <release>/acoustic.safetensors + acoustic.json   {"format", "model_config", "speakers": [voice]}
    <release>/decoder.safetensors  + decoder.json    {"format", "model_config"}
    <release>/onnx/                                    graphs for ONNX Runtime and browsers (okur.export_web)

`read` returns the same dictionaries as a training checkpoint, so every runtime accepts either.
"""

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

FORMAT = "okur-release-1"


def read(path: Path) -> dict[str, Any]:
    """A training checkpoint (.pt, trusted local file) or a release model (.safetensors with its .json)."""
    if path.suffix != ".safetensors":
        return torch.load(path, map_location="cpu", weights_only=False)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: unknown format {meta.get('format')!r}")
    weights = load_file(path)
    if "speakers" in meta:
        return {"model_config": meta["model_config"], "average": weights, "speakers": meta["speakers"]}
    return {"model_config": meta["model_config"], "model": weights}


def _save(weights: dict[str, torch.Tensor], meta: dict[str, Any], path: Path) -> None:
    save_file({k: v.detach().contiguous() for k, v in weights.items()}, path, metadata={"format": FORMAT})
    path.with_suffix(".json").write_text(json.dumps({"format": FORMAT, **meta}, indent=1, ensure_ascii=False) + "\n")


def write(acoustic: Path, decoder: Path, voice: str, out_dir: Path) -> list[Path]:
    """Keep one voice of `acoustic` (its averaged weights) and the decoder's generator; drop optimizer state."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = read(acoustic)
    index = state["speakers"].index(voice)
    weights = dict(state["average"])
    weights["speaker.weight"] = weights["speaker.weight"][index:index + 1].clone()
    config = {**state["model_config"], "n_speakers": 1}
    _save(weights, {"model_config": config, "speakers": [voice]}, out_dir / "acoustic.safetensors")
    dstate = read(decoder)
    _save(dstate["model"], {"model_config": dstate["model_config"]}, out_dir / "decoder.safetensors")
    return [out_dir / f for f in ("acoustic.safetensors", "acoustic.json", "decoder.safetensors", "decoder.json")]
