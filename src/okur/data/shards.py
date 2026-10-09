"""Prepared training data: Parquet shards of (text, letters, letter durations, latents), written atomically."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from jaxtyping import Float16, Float32

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("source", pa.string()),
    ("speaker", pa.string()),
    ("quality", pa.int8()),
    ("text", pa.string()),
    ("letters", pa.string()),
    ("dur", pa.list_(pa.float32())),  # latent frames per letter; sums to n_frames
    ("n_frames", pa.int32()),
    ("latents", pa.binary()),  # float16, (n_frames, 64), row-major
    ("score", pa.float32()),
    ("audio", pa.binary()),  # the cropped clip, 48 kHz mono FLAC: decoder training target
])


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    speaker: str
    quality: int
    text: str
    letters: str
    dur: Float32[np.ndarray, "..."]
    latents: Float16[np.ndarray, "..."]
    score: float
    audio: bytes = b""


def shard_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"shard-{index:05d}.parquet"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temporary file, flush to disk, then rename: readers see the old file or the new one, never half."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_shard(out_dir: Path, index: int, examples: list[Example], stats: dict[str, int]) -> None:
    table = pa.Table.from_pylist([{
        **{k: v for k, v in asdict(e).items() if k not in ("dur", "latents")},
        "dur": e.dur.tolist(), "n_frames": len(e.latents), "latents": e.latents.tobytes(),
    } for e in examples], schema=SCHEMA)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    atomic_write_bytes(out_dir / f"shard-{index:05d}.json", json.dumps(stats, indent=1).encode())
    atomic_write_bytes(shard_path(out_dir, index), sink.getvalue().to_pybytes())  # last: marks the shard done


def read_shards(paths: list[Path], *, audio: bool = False) -> list[Example]:
    """Load shards; the audio column (large) only when asked for."""
    columns = [n for n in SCHEMA.names if audio or n != "audio"]
    out: list[Example] = []
    for path in paths:
        for row in pq.read_table(path, columns=columns).to_pylist():
            latents = np.frombuffer(row["latents"], dtype=np.float16).reshape(row["n_frames"], 64)
            out.append(Example(row["id"], row["source"], row["speaker"], row["quality"], row["text"], row["letters"],
                               np.asarray(row["dur"], dtype=np.float32), latents, row["score"],
                               row.get("audio") or b""))
    return out
