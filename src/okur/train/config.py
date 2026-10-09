from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from okur.model.config import AcousticConfig


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_dir: Path
    data: list[Path]  # directories of prepared shards
    model: AcousticConfig = AcousticConfig()
    init_from: Path | None = None  # fine-tune: start from this checkpoint's averaged weights and speaker table
    only_speakers: list[str] | None = None  # train on these speakers only (e.g. one target voice)

    # optimisation
    steps: Annotated[int, Field(gt=0)] = 100_000
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    warmup: int = 2_000
    weight_decay: float = 0.01
    optimizer: Literal["adamw", "muon"] = "adamw"
    grad_clip: float = 1.0
    ema_decay: float = 0.999  # weight averaging for the sampled / exported model
    cond_drop: float = 0.1
    dur_weight: float = 1.0

    # batching
    max_frames: Annotated[int, Field(gt=0)] = 8_000  # latent frames per batch per device; 8000 = 320 s of audio
    max_letters: int = 6_000
    workers: int = 4
    seed: int = 0
    pad_frames: int = 1  # pad batch lengths to multiples, bounding the shapes torch.compile / CUDA graphs see
    pad_letters: int = 1
    latents_on_device: bool = False  # keep all latents on the GPU; batches carry indices only

    # hardware
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    precision: Literal["fp32", "bf16"] = "fp32"
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune-no-cudagraphs"] = "default"

    # fault tolerance
    ckpt_every_steps: int = 1_000
    ckpt_every_minutes: float = 15.0
    keep_checkpoints: int = 3
    max_bad_steps: int = 20  # consecutive non-finite steps before rolling back to the last checkpoint

    # monitoring
    log_every: int = 50
    sample_every: int = 2_000
    sample_speaker: str | None = None  # speaker used for samples/CER (default: the first in the table)
    sample_texts: list[str] = ["Bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor.", "Ofisimiz 3. katta."]
    sample_steps: int = 16
    early_stop_patience: int = 0  # stop after this many CER evaluations without improvement (0: never)
    early_stop_min_delta: float = 0.005
    guidance: float = 3.0  # Phase 0: 3.0 gave CER 0.142 vs 0.227 at 2.0

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        return cls.model_validate(yaml.safe_load(path.read_text()))
