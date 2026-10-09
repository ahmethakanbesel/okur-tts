#!/usr/bin/env bash
# Wait until the Common Voice terms are accepted on Mozilla Data Collective, then download, extract and prepare the
# corpus (alongside any other preparation), then release the training hold (runs/HOLD). Resumable: re-run any time.
#
#   source /root/.tts_secrets && scripts/fetch_cv.sh     # needs MDC_API_KEY
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ID="${CV_DATASET_ID:-cmu5wkah500c2o10719lem8m8}"
RAW=data/raw/cv
mkdir -p "$RAW" data/prepared

until curl -s -X POST -H "Authorization: Bearer ${MDC_API_KEY:?}" \
    "https://mozilladatacollective.com/api/datasets/$ID/download" | grep -qv '"error"'; do
  echo "$(date '+%F %T') Common Voice terms not accepted yet; checking again in 60 s"; sleep 60
done
echo "$(date '+%F %T') terms accepted"

if [[ ! -f "$RAW/.extracted" ]]; then
  archive=$(cd "$RAW" && uv run --no-project --with datacollective python -c "from datacollective import download_dataset
print(download_dataset('$ID', download_directory='.', show_progress=False))" | tail -1)
  (cd "$RAW" && tar xf "$archive" && rm -f "$archive" && touch .extracted) || { echo "extraction failed" >&2; exit 1; }
fi
root="$(dirname "$(find "$RAW" -name validated.tsv -path '*/tr/*' | head -1)")"
echo "$(date '+%F %T') Common Voice at $root; preparing"
uv run okur prepare commonvoice_tr "$root" data/prepared/commonvoice_tr --device cuda --workers 12 \
  --align-batch 16 --encode-seconds 40 >> data/prepared/commonvoice_tr.log 2>&1 && rm -f runs/HOLD \
  && echo "$(date '+%F %T') Common Voice prepared; training hold released"
