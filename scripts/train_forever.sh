#!/usr/bin/env bash
# Keep a training run alive: restart after crashes, OOM kills, or pre-emption until the run writes DONE.
# Training resumes exactly from the newest checkpoint each time.
#
#   scripts/train_forever.sh configs/mac_smoke.yaml                      # one device (Mac, CPU, or a single GPU)
#   GPUS=4 scripts/train_forever.sh configs/gpu_teacher.yaml             # one process per GPU (DDP)
#   KIND=decoder GPUS=2 scripts/train_forever.sh configs/gpu_decoder.yaml  # the decoder GAN instead
#
# A hang watchdog restarts the run if the heartbeat file stops updating for HANG_MINUTES.
set -uo pipefail

config="$1"
gpus="${GPUS:-1}"
max_restarts="${MAX_RESTARTS:-50}"
hang_minutes="${HANG_MINUTES:-20}"
case "${KIND:-acoustic}" in
  acoustic) command=train; module=okur.train.trainer ;;
  decoder) command=train-decoder; module=okur.train.decoder_trainer ;;
  distill) command=distill; module=okur.train.distill ;;
  *) echo "KIND must be acoustic, decoder or distill" >&2; exit 2 ;;
esac
run_dir="$(uv run python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['run_dir'])" "$config")"
mkdir -p "$run_dir"

# Hold: while runs/HOLD exists (e.g. a corpus is still being prepared), wait before training starts.
while [[ -f runs/HOLD ]]; do
  echo "$(date '+%F %T') waiting: runs/HOLD exists ($(cat runs/HOLD 2>/dev/null))"; sleep 60
done

for attempt in $(seq 1 "$max_restarts"); do
  [[ -f "$run_dir/DONE" ]] && { echo "run complete: $run_dir"; exit 0; }
  echo "$(date '+%F %T') attempt $attempt" | tee -a "$run_dir/restarts.log"
  if [[ "$gpus" -gt 1 ]]; then
    uv run torchrun --standalone --nproc_per_node="$gpus" -m "$module" "$config" &
  else
    uv run okur "$command" "$config" &
  fi
  pid=$!
  started=$(date +%s)
  # Watchdog: if neither the heartbeat nor this attempt's start is recent, the run is hung: SIGTERM (the trainer
  # saves and exits), then SIGKILL.
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5  # notice an exit within seconds: idle rented GPUs cost money
    hb="$run_dir/heartbeat.json"
    last=$started
    [[ -f "$hb" ]] && last=$(( $(date -r "$hb" +%s) > started ? $(date -r "$hb" +%s) : started ))
    if [[ $(( $(date +%s) - last )) -gt $(( hang_minutes * 60 )) ]]; then
      echo "$(date '+%F %T') heartbeat stale; restarting" | tee -a "$run_dir/restarts.log"
      kill -TERM "$pid"; sleep 120; kill -KILL "$pid" 2>/dev/null
    fi
  done
  wait "$pid"; code=$?
  echo "$(date '+%F %T') exited with $code" | tee -a "$run_dir/restarts.log"
  sleep 10
done
echo "gave up after $max_restarts restarts" >&2
exit 1
