"""Corpus readers list deterministically and load audio, transcript and speaker."""

import io
from pathlib import Path

import numpy as np
import soundfile as sf

from okur.data.sources import CommonVoice, Fleurs, PairedFiles


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(1600, np.float32), 16000)


def test_common_voice(tmp_path: Path) -> None:
    rows = ["client_id\tpath\tsentence"] + [f"{'a' if i < 300 else 'b'}\tc{i:04d}.wav\tcümle {i}" for i in range(305)]
    (tmp_path / "validated.tsv").write_text("\n".join(rows), encoding="utf-8")
    for i in (0, 304):
        _wav(tmp_path / "clips" / f"c{i:04d}.wav")
    cv = CommonVoice(tmp_path)
    refs = cv.refs()
    assert len(refs) == 305 and refs == sorted(refs, key=lambda r: r.file)
    first, last = cv.load([refs[0], refs[-1]])
    assert (first.speaker, last.speaker) == ("cv:a", "cv:other")  # b has too few clips
    assert first.text == "cümle 0" and sf.read(io.BytesIO(first.audio))[1] == 16000


def test_fleurs_and_paired(tmp_path: Path) -> None:
    (tmp_path / "f").mkdir()
    (tmp_path / "f" / "train.tsv").write_text("1\tx.wav\tMerhaba Dünya.\tmerhaba dünya\tm e r\t1600\tMALE\n")
    (tmp_path / "f" / "dev.tsv").write_text("")
    _wav(tmp_path / "f" / "train" / "x.wav")
    (item,) = Fleurs(tmp_path / "f").load(Fleurs(tmp_path / "f").refs())
    assert item.text == "Merhaba Dünya."
    _wav(tmp_path / "p" / "Train" / "1.wav")
    (tmp_path / "p" / "Train" / "1.txt").write_text("bir iki\n")
    _wav(tmp_path / "p" / "Train" / "2.wav")  # no transcript: skipped
    src = PairedFiles(tmp_path / "p")
    (item,) = src.load(src.refs())
    assert item.text == "bir iki" and item.speaker == "issai_tsc"
