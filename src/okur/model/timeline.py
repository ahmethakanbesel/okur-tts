"""The word–letter timeline: which word, and where inside it, every letter and every output frame sits.

A word owns its letters and the spaces and punctuation that follow it, so pauses belong to the word before them.
Training builds the timeline from aligned durations; inference from predicted ones. Both use these functions.
"""

from dataclasses import dataclass
from itertools import pairwise

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


@dataclass(frozen=True)
class Words:
    cw: Int[Tensor, " letters"]  # word of each letter
    wstart: Int[Tensor, " letters"]  # first letter of that word
    n_words: int


def words(letters: str) -> Words:
    starts = [i for i, ch in enumerate(letters) if ch != " " and (i == 0 or letters[i - 1] == " ")] or [0]
    bounds = [0, *starts[1:], len(letters)]
    cw: list[int] = []
    wstart: list[int] = []
    for w, (a, b) in enumerate(pairwise(bounds)):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return Words(torch.tensor(cw), torch.tensor(wstart), len(bounds) - 1)


def word_frames(dur: Float[Tensor, " letters"], cw: Int[Tensor, " letters"], n_words: int) -> Int[Tensor, " words"]:
    """Whole frames per word, rounded on the running total so the sum never drifts from the sum of durations."""
    per_word = torch.zeros(n_words, dtype=torch.float64).scatter_add_(0, cw, dur.double())
    ends = per_word.cumsum(0).round().long()
    return torch.diff(ends, prepend=ends.new_zeros(1)).clamp(min=0)


def frame_timeline(counts: Int[Tensor, " words"]) -> tuple[Int[Tensor, " frames"], Float[Tensor, " frames"]]:
    """Word of each frame and its position inside that word, in [0, 1)."""
    fw = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    start = (counts.cumsum(0) - counts)[fw]
    fp = ((torch.arange(fw.numel()) - start).double() / counts[fw].double()).float()
    return fw, fp


def letter_positions(dur: Float[Tensor, "b l"], mask: Bool[Tensor, "b l"], cw: Int[Tensor, "b l"],
                     wstart: Int[Tensor, "b l"]) -> Float[Tensor, "b l"]:
    """Centre of each letter inside its word, in [0, 1], from (predicted or aligned) letter durations."""
    m = mask.float()
    c = dur.clamp(min=1e-4) * m
    done = c.cumsum(-1)
    before = done - c
    word = cw.clamp(min=0)
    total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
    return ((done - before.gather(1, wstart) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * m
