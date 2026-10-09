"""Score every wav under runs/utmos/<system>/ with UTMOS22 (strong learner). Runs in an isolated Python 3.10 env:

    uv run --no-project --python 3.10 --with utmos --with soundfile --with "numpy<2" \
        python scripts/utmos_score.py runs/utmos

Prints mean ± 95% confidence interval per system and writes runs/utmos/utmos.json.

The UTMOS checkpoint (hf://mosnets/utmos) is a pickle that torch's weights-only loader cannot read (it contains a
defaultdict). Before loading it unrestricted, its pickle is audited: every global it imports must be in ALLOWED (data
containers and torch's tensor rebuilders only). Anything else aborts.
"""

import contextlib
import io
import json
import math
import pickletools
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import soundfile as sf
import torch
import utmos
from cached_path import cached_path

CHECKPOINT = "hf://mosnets/utmos/model.ckpt"
ALLOWED = {
    "__builtin__.dict", "__builtin__.list", "__builtin__.long", "builtins.dict", "builtins.list", "builtins.int",
    "collections.OrderedDict", "collections.defaultdict", "typing.Any",
    "omegaconf.base.ContainerMetadata", "omegaconf.base.Metadata", "omegaconf.dictconfig.DictConfig",
    "omegaconf.listconfig.ListConfig", "omegaconf.nodes.AnyNode",
    "torch.DoubleStorage", "torch.FloatStorage", "torch.LongStorage", "torch._utils._rebuild_tensor_v2",
}


def pickle_globals(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as z:
        data = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    ops = list(pickletools.genops(io.BytesIO(data)))
    found: set[str] = set()
    for i, (op, arg, _) in enumerate(ops):
        if op.name == "GLOBAL":
            found.add(str(arg).replace(" ", "."))
        elif op.name == "STACK_GLOBAL":
            strings = [a for (o, a, _) in ops[max(0, i - 4):i] if o.name in ("SHORT_BINUNICODE", "BINUNICODE")]
            found.add(f"{strings[-2]}.{strings[-1]}")
    return found


@contextlib.contextmanager
def audited_full_load() -> Iterator[None]:
    unexpected = pickle_globals(Path(cached_path(CHECKPOINT))) - ALLOWED
    if unexpected:
        raise RuntimeError(f"UTMOS checkpoint imports unexpected objects, refusing to load: {sorted(unexpected)}")
    original = torch.load

    def load(*args, **kwargs):  # the audit above is the safety check
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def main(root: Path) -> None:
    with audited_full_load():
        scorer = utmos.Score()
    results = {}
    for system in sorted(p for p in root.iterdir() if p.is_dir()):
        scores, per_file = [], {}
        for wav in sorted(system.glob("*.wav")):
            audio, sr = sf.read(wav, dtype="float32", always_2d=True)
            mono = torch.from_numpy(audio.mean(1))[None]
            scores.append(float(scorer.calculate_wav(mono, sr)))
            per_file[wav.stem] = scores[-1]
        n = len(scores)
        mean = sum(scores) / n
        sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / max(1, n - 1))
        results[system.name] = {"utmos": mean, "ci95": 1.96 * sd / math.sqrt(n), "n": n, "per_file": per_file}
        print(f"{system.name:16s} UTMOS {mean:.3f} ± {1.96 * sd / math.sqrt(n):.3f}  (n={n})", flush=True)
    (root / "utmos.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
