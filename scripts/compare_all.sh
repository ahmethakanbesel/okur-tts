#!/usr/bin/env bash
# The full system comparison for the web demo, on a CUDA machine (one RTX 5090: ~1 h). Writes demo/metrics.json,
# demo/samples.json and demo/audio/. Speed is measured with every system pinned to the same 8 CPU cores.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
[ -f /root/.tts_secrets ] && { set -a; . /root/.tts_secrets; set +a; }
step() { echo "$(date +%T) $*"; }
# FROM=<stage> resumes at a stage: sentences, render, speed, asr, utmos, web
FROM="${FROM:-sentences}"
stages=(sentences render speed asr utmos web)
at() { local i; for i in "${stages[@]}"; do [ "$i" = "$FROM" ] && return 0; [ "$i" = "$1" ] && return 1; done; }
uv sync -q --group export

PIPER_MODEL=runs/third_party/piper/dfki/tr_TR-dfki-medium.onnx
MMS_MODEL=runs/third_party/mms_tur
piper() { uv run --no-project --with piper-tts --with soundfile python -I scripts/render_third_party.py piper "$PIPER_MODEL" "$@"; }
mms() { uv run --no-project --with "transformers<5" --with torch --with soundfile python -I scripts/render_third_party.py mms "$MMS_MODEL" "$@"; }
utmos() { uv run --no-project --python 3.10 --with utmos --with soundfile --with "numpy<2" python scripts/utmos_score.py "$@"; }
compare() { uv run python scripts/compare_systems.py "$@"; }

if at sentences; then
step "sentences and third-party models"
compare export
uv run python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
d = Path("runs/third_party/piper/dfki")
for f in ("tr_TR-dfki-medium.onnx", "tr_TR-dfki-medium.onnx.json", "MODEL_CARD"):
    hf_hub_download("rhasspy/piper-voices", f"tr/tr_TR/dfki/medium/{f}", local_dir="runs/third_party/piper-voices")
    (d / f).parent.mkdir(parents=True, exist_ok=True)
    (d / f).write_bytes(Path("runs/third_party/piper-voices/tr/tr_TR/dfki/medium", f).read_bytes())
snapshot_download("facebook/mms-tts-tur", local_dir="runs/third_party/mms_tur")
PY
fi

if at render; then
step "render: ours + EMA Lightning (GPU)"
compare render
step "render: Piper, MMS"
piper demo/hard_sentences.json runs/hard/piper_dfki
piper runs/compare/freya_sentences.json runs/utmos/piper_dfki
mms demo/hard_sentences.json runs/hard/mms_tur
mms runs/compare/freya_sentences.json runs/utmos/mms_tur
fi

if at speed; then
step "speed: 8 pinned CPU cores each"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
taskset -c 0-7 uv run python scripts/compare_systems.py bench ours
taskset -c 0-7 uv run python scripts/compare_systems.py bench ema_lightning
taskset -c 0-7 bash -c "$(declare -f piper); PIPER_MODEL=$PIPER_MODEL; piper demo/hard_sentences.json runs/bench/piper_dfki"
taskset -c 0-7 bash -c "$(declare -f mms); MMS_MODEL=$MMS_MODEL; mms demo/hard_sentences.json runs/bench/mms_tur"
unset OMP_NUM_THREADS MKL_NUM_THREADS
fi

if at asr; then
step "intelligibility: Whisper-large-v3"
compare asr hard ours ema_lightning piper_dfki mms_tur
compare asr freya v02 v02_dec110k ema_lightning piper_dfki mms_tur
fi
if at utmos; then
step "naturalness: UTMOS"
utmos runs/utmos
utmos runs/hard
fi
step "web audio and metrics"
compare mp3
compare metrics
step "comparison done"
