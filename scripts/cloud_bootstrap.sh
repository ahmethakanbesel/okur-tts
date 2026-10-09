#!/usr/bin/env bash
# Set up a fresh rented GPU machine (Vast.ai / RunPod, Ubuntu + NVIDIA driver) and run the whole pipeline.
#
#   rsync -az --exclude .venv --exclude data --exclude runs ./ root@HOST:/workspace/tts/   # from the Mac
#   ssh root@HOST 'cd /workspace/tts && HF_TOKEN=hf_... MDC_API_KEY=... scripts/cloud_bootstrap.sh'
#
# MDC_API_KEY: Mozilla Data Collective API key (Profile → API), after accepting the Common Voice dataset's terms.
#
# Every stage is resumable: re-running this script after a disconnect or a lost instance continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS="${GPUS:-$(nvidia-smi -L | wc -l)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # several processes share the GPU: avoid fragmentation
RAW=data/raw
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN (huggingface.co/settings/tokens): anonymous downloads are rate limited}"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --no-default-groups --group data --group dev

# 1. Download every corpus in parallel (each skips files it already has).
mkdir -p "$RAW"
hf() { uv run hf download --repo-type dataset "$@"; }
( hf ymoslem/MediaSpeech --include "tr/*" --local-dir "$RAW/mediaspeech" && ln -sfn mediaspeech/tr "$RAW/mediaspeech_tr" ) &
( hf google/fleurs --include "data/tr_tr/*" --local-dir "$RAW/fleurs" && cd "$RAW/fleurs/data/tr_tr" \
  && for s in train dev; do [ -d "$s" ] || tar xzf "audio/$s.tar.gz"; done ) &
( cd "$RAW" && { [ -f tsc/.extracted ] || { hf issai/Turkish_Speech_Corpus --local-dir tsc && cd tsc \
  && tar xzf ISSAI_TSC_218.tar.gz && rm ISSAI_TSC_218.tar.gz && touch .extracted; }; } ) &  # free 21 GB of disk
CV_DATASET_ID="${CV_DATASET_ID:-cmu5wkah500c2o10719lem8m8}"  # Common Voice Scripted Speech, Turkish (CC0)
if [[ -n "${MDC_API_KEY:-}" ]]; then
  ( mkdir -p "$RAW/cv" && cd "$RAW/cv" && { [ -f .extracted ] || {
      archive=$(uv run --no-project --with datacollective python -c "from datacollective import download_dataset; \
print(download_dataset('$CV_DATASET_ID', download_directory='.', show_progress=False))" | tail -1) \
      && tar xf "$archive" && rm -f "$archive" && touch .extracted; }; } ) &
fi
wait
CV_ROOT="$(dirname "$(find "$RAW/cv" -name validated.tsv -path '*/tr/*' 2>/dev/null | head -1)" 2>/dev/null || true)"

# 2. Prepare shards: one process per GPU per corpus, every GPU busy; CPU workers split across them.
workers=$(( $(nproc) / GPUS ))
prep() {  # source raw_root
  for r in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$r uv run okur prepare "$1" "$2" "data/prepared/$1" --device cuda --rank "$r" \
      --world "$GPUS" --workers "$workers" --align-batch 16 --encode-seconds 40 >> "data/prepared/$1.log" 2>&1 &
  done
  wait
}
mkdir -p data/prepared
prep mediaspeech_tr "$RAW/mediaspeech_tr"
prep fleurs_tr "$RAW/fleurs/data/tr_tr"
prep issai_tsc "$RAW/tsc"
[[ -n "$CV_ROOT" ]] && prep commonvoice_tr "$CV_ROOT"

# 3. Measure every speed option on this machine (~10 min); results in runs/bench/bench.json. Adjust the gpu_*.yaml
#    speed settings from the table if a variant beats the defaults.
[[ -f runs/bench/bench.json ]] || uv run okur bench data/prepared/*/ --out runs/bench | tee runs-bench.log

# 4. Train. The decoder and the teacher are independent and neither fills a modern GPU alone, so they always run at the
#    same time: on separate GPUs when there are 2+, sharing the GPU otherwise (SEQUENTIAL=1 to run one after the other).
if [[ "$GPUS" -ge 2 ]]; then
  half=$(( GPUS / 2 ))
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((half - 1))) GPUS=$half \
    scripts/train_forever.sh configs/gpu_teacher.yaml > runs-teacher.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$(seq -s, $half $((GPUS - 1))) GPUS=$((GPUS - half)) KIND=decoder \
    scripts/train_forever.sh configs/gpu_decoder.yaml > runs-decoder.log 2>&1 &
  wait
elif [[ -n "${SEQUENTIAL:-}" ]]; then
  scripts/train_forever.sh configs/gpu_teacher.yaml
  KIND=decoder scripts/train_forever.sh configs/gpu_decoder.yaml
else
  scripts/train_forever.sh configs/gpu_teacher.yaml > runs-teacher.log 2>&1 &
  KIND=decoder scripts/train_forever.sh configs/gpu_decoder.yaml > runs-decoder.log 2>&1 &
  wait
fi
