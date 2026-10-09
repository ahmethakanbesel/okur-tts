"""Raw corpora, listed deterministically so every machine splits the work into the same shards.

Every source lists RawRefs (cheap, sorted) and loads RawItems (audio bytes + transcript + speaker) for a shard's refs.
"""

import csv
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True)
class RawRef:
    source: str
    file: str
    row: int


@dataclass(frozen=True)
class RawItem:
    id: str
    source: str
    speaker: str
    text: str
    audio: bytes


class MediaSpeech:
    """MediaSpeech Turkish (CC-BY 4.0): Parquet files of (audio, sentence), no speaker labels."""

    name = "mediaspeech_tr"

    def __init__(self, root: Path) -> None:
        self.files = sorted(root.glob("*.parquet"))

    def refs(self) -> list[RawRef]:
        return [RawRef(self.name, str(f), i) for f in self.files for i in range(pq.read_metadata(f).num_rows)]

    def load(self, refs: list[RawRef]) -> Iterator[RawItem]:
        by_file: dict[str, list[int]] = {}
        for r in refs:
            by_file.setdefault(r.file, []).append(r.row)
        for file, rows in by_file.items():
            table = pq.read_table(file).take(rows)
            for row_id, rec in zip(rows, table.to_pylist(), strict=True):
                yield RawItem(f"{Path(file).stem}-{row_id}", self.name, self.name, rec["sentence"],
                              rec["audio"]["bytes"])


class CommonVoice:
    """Mozilla Common Voice (CC0), extracted: <root>/validated.tsv and <root>/clips/*.mp3. Speakers with fewer than
    MIN_CLIPS clips are pooled into one speaker, so the speaker table holds only voices the model can learn."""

    name = "commonvoice_tr"
    MIN_CLIPS = 300

    def __init__(self, root: Path) -> None:
        self.root = root
        with open(root / "validated.tsv", newline="", encoding="utf-8") as f:
            rows = sorted(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE), key=lambda r: r["path"])
        counts = Counter(r["client_id"] for r in rows)
        self.rows = {r["path"]: (r["sentence"], r["client_id"] if counts[r["client_id"]] >= self.MIN_CLIPS else "other")
                     for r in rows}

    def refs(self) -> list[RawRef]:
        return [RawRef(self.name, path, 0) for path in self.rows]

    def load(self, refs: list[RawRef]) -> Iterator[RawItem]:
        for r in refs:
            text, speaker = self.rows[r.file]
            yield RawItem(Path(r.file).stem, self.name, f"cv:{speaker[:12]}", text,
                          (self.root / "clips" / r.file).read_bytes())


class Fleurs:
    """Google FLEURS Turkish (CC-BY 4.0), extracted: <root>/<split>.tsv and <root>/<split>/*.wav. No speaker labels."""

    name = "fleurs_tr"

    def __init__(self, root: Path, splits: tuple[str, ...] = ("train", "dev")) -> None:
        self.root = root
        self.rows: dict[str, str] = {}
        for split in splits:
            with open(root / f"{split}.tsv", newline="", encoding="utf-8") as f:
                for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                    self.rows[f"{split}/{row[1]}"] = row[2]  # raw transcription, with casing and punctuation

    def refs(self) -> list[RawRef]:
        return [RawRef(self.name, path, 0) for path in sorted(self.rows)]

    def load(self, refs: list[RawRef]) -> Iterator[RawItem]:
        for r in refs:
            yield RawItem(r.file.replace("/", "-"), self.name, self.name, self.rows[r.file],
                          (self.root / r.file).read_bytes())


class PairedFiles:
    """Any corpus laid out as audio files with a same-named .txt transcript beside them (e.g. ISSAI TSC, MIT).
    One speaker label for the whole corpus."""

    name = "paired"
    AUDIO = (".wav", ".flac", ".mp3", ".ogg")

    def __init__(self, root: Path, name: str = "issai_tsc") -> None:
        self.root, self.name = root, name
        self.files = sorted(p for p in root.rglob("*")
                            if p.suffix.lower() in self.AUDIO and p.with_suffix(".txt").exists())

    def refs(self) -> list[RawRef]:
        return [RawRef(self.name, str(p.relative_to(self.root)), 0) for p in self.files]

    def load(self, refs: list[RawRef]) -> Iterator[RawItem]:
        for r in refs:
            path = self.root / r.file
            yield RawItem(r.file.replace("/", "-"), self.name, self.name,
                          path.with_suffix(".txt").read_text(encoding="utf-8").strip(), path.read_bytes())


SOURCES = {MediaSpeech.name: MediaSpeech, CommonVoice.name: CommonVoice, Fleurs.name: Fleurs, "issai_tsc": PairedFiles}
