"""Checkpoints that survive crashes, pre-emption and full disks.

- Saved from a CPU copy in a background thread, so training does not wait for the disk.
- Written to a temporary file, flushed and renamed: a crash mid-save leaves the previous checkpoint intact.
- Loading tries the newest checkpoint first and falls back to older ones if a file is unreadable.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class Checkpointer:
    def __init__(self, run_dir: Path, keep: int) -> None:
        self.dir = run_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        for stale in self.dir.glob(".*.tmp"):  # left by a crash during a save
            stale.unlink(missing_ok=True)

    def paths(self) -> list[Path]:
        return sorted(self.dir.glob("step-*.pt"), reverse=True)

    def save(self, step: int, state: dict[str, Any], *, wait: bool = False, name: str | None = None) -> None:
        """Save `state` as step-<step>.pt (rotated), or as `name` (e.g. best.pt, kept until overwritten)."""
        self.wait()  # one save in flight at a time
        snapshot = _to_cpu(state)
        path = self.dir / (name or f"step-{step:08d}.pt")
        self._thread = threading.Thread(target=self._write, args=(path, snapshot), daemon=False)
        self._thread.start()
        if wait:
            self.wait()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("checkpoint save failed") from error

    def _write(self, path: Path, state: dict[str, Any]) -> None:
        try:
            tmp = self.dir / f".{path.name}.tmp"
            with open(tmp, "wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            for old in self.paths()[self.keep:]:
                old.unlink(missing_ok=True)
            log.info("saved %s", path.name)
        except BaseException as e:  # surfaced on the next save or wait
            self._error = e

    def load_latest(self) -> dict[str, Any] | None:
        for path in self.paths():
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
                if "step" in state and "model" in state:
                    log.info("resuming from %s", path.name)
                    return state
            except Exception as e:
                log.warning("unreadable checkpoint %s (%s); trying an older one", path.name, e)
        return None
