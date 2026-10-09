"""Letter timings from a Turkish CTC speech recognizer, by forced alignment.

The recognizer (wav2vec2 XLS-R fine-tuned on Common Voice Turkish, CC-BY 4.0) emits letter probabilities every 20 ms.
Viterbi alignment of the known transcript through those probabilities gives every letter a start time. Letters the
recognizer has no symbol for (punctuation) get zero length; â, î and û align as a, i and u.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jaxtyping import Float32, Float64, Int64

ALIGNER_REPO = "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish"
CTC_RATE = 50  # emission frames per second
_FOLD = str.maketrans({"â": "a", "î": "i", "û": "u", " ": "|"})


@dataclass(frozen=True)
class Alignment:
    starts: Float64[np.ndarray, "..."]  # seconds, one per letter of the transcript
    end: float  # seconds, end of the last aligned letter
    score: float  # mean log-probability of the aligned letters: low means the transcript does not match


class Recognizer:
    """Batched CTC emissions on the given device."""

    def __init__(self, device: str, *, bf16: bool = False) -> None:
        """bf16 (CUDA only) roughly doubles throughput; alignment needs only the argmax-level shape of emissions."""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.bf16 = bf16 and device.startswith("cuda")

        self.processor: Any = Wav2Vec2Processor.from_pretrained(ALIGNER_REPO)
        model: Any = Wav2Vec2ForCTC.from_pretrained(ALIGNER_REPO)
        self.model: Any = model.to(device).eval()
        self.device = device
        self.vocab: dict[str, int] = self.processor.tokenizer.get_vocab()
        self.blank = self.processor.tokenizer.pad_token_id

    @torch.inference_mode()
    def emissions(self, batch: list[Float32[np.ndarray, "..."]]) -> list[Float32[np.ndarray, "..."]]:
        """Log-probabilities per 20 ms frame for each 16 kHz clip."""
        inputs = self.processor(batch, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            logits = self.model(inputs.input_values.to(self.device),
                                attention_mask=inputs.attention_mask.to(self.device)).logits
        logp = torch.log_softmax(logits.float(), -1).cpu().numpy()
        lengths = self.model._get_feat_extract_output_lengths(inputs.attention_mask.sum(-1)).tolist()
        return [logp[i, :n] for i, n in enumerate(lengths)]

    def tokens(self, letters: str) -> tuple[list[int], list[int]]:
        """CTC token ids for the letters it knows, and which letter each token came from."""
        ids: list[int] = []
        where: list[int] = []
        for i, ch in enumerate(letters.translate(_FOLD)):
            if ch in self.vocab:
                ids.append(self.vocab[ch])
                where.append(i)
        return ids, where


def viterbi(logp: Float32[np.ndarray, "..."], tokens: list[int],
            blank: int) -> tuple[Int64[np.ndarray, "..."], int, float] | None:
    """Most likely CTC path for `tokens`: first frame of every token, last frame of the last token, mean token
    log-probability. None if the clip is too short for the transcript."""
    n_frames, n = len(logp), len(tokens)
    ext = np.full(2 * n + 1, blank, dtype=np.int64)
    ext[1::2] = tokens
    s = len(ext)
    skip = np.zeros(s, dtype=bool)
    skip[3::2] = ext[3::2] != ext[1:-2:2]  # may jump over a blank between two different tokens
    if n == 0 or n_frames < n + int((~skip[3::2]).sum()):
        return None
    emit = logp[:, ext]
    score = np.full(s, -np.inf)
    score[0], score[1] = emit[0, 0], emit[0, 1]
    back = np.zeros((n_frames, s), dtype=np.int8)
    for t in range(1, n_frames):
        one = np.concatenate(([-np.inf], score[:-1]))
        two = np.where(skip, np.concatenate(([-np.inf, -np.inf], score[:-2])), -np.inf)
        stacked = np.stack([score, one, two])
        back[t] = stacked.argmax(0)
        score = stacked.max(0) + emit[t]
    state = s - 1 if score[s - 1] >= score[s - 2] else s - 2
    if not np.isfinite(score[state]):
        return None
    path = np.empty(n_frames, dtype=np.int64)
    for t in range(n_frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    frames = np.nonzero(path % 2 == 1)[0]
    token = path[frames] // 2
    first = np.full(n, -1, dtype=np.int64)
    first[token[::-1]] = frames[::-1]  # the earliest frame wins
    if (first < 0).any():
        return None
    last = int(frames[token == n - 1].max())
    return first, last, float(emit[frames, path[frames]].mean())


def align(logp: Float32[np.ndarray, "..."], letters: str, tokens: list[int], where: list[int],
          blank: int) -> Alignment | None:
    found = viterbi(logp, tokens, blank)
    if found is None:
        return None
    first, last, score = found
    end = (last + 1) / CTC_RATE
    # A letter without a CTC symbol (punctuation) starts where the next aligned letter starts.
    nxt = np.searchsorted(np.asarray(where), np.arange(len(letters)), side="left")
    token_start = first / CTC_RATE
    starts = np.where(nxt < len(tokens), token_start[np.minimum(nxt, len(tokens) - 1)], end)
    return Alignment(starts, end, score)
