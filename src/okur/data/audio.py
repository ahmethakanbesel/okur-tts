import io

import numpy as np
import soundfile as sf
import soxr
from jaxtyping import Float32

Audio = Float32[np.ndarray, "..."]

# Recording-quality buckets, fed to the model as a condition so inference can ask for clean full-band speech.
FULL_BAND, SUPER_WIDE, WIDE, NARROW = 0, 1, 2, 3


def decode(data: bytes) -> tuple[Audio, int]:
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(1), int(rate)


def resample(audio: Audio, rate: int, target: int) -> Audio:
    return audio if rate == target else soxr.resample(audio, rate, target, quality="HQ").astype(np.float32)


def quality_bucket(rate: int) -> int:
    """Bucket by the source's sample rate: what bandwidth the original recording can carry at most."""
    if rate >= 44100:
        return FULL_BAND
    if rate >= 22050:
        return SUPER_WIDE
    if rate >= 16000:
        return WIDE
    return NARROW
