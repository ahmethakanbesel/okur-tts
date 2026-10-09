#!/usr/bin/env bash
# Assemble the public repository from this working tree: an explicit list of what is published. The first run creates
# a fresh git history; later runs replace the files in the existing repo and commit the difference (push yourself).
# Internal notes (PLAN.md, RUNBOOK.md), experiment one-offs, research audio and third-party demo assets stay here.
#   scripts/make_public.sh ../okur ["commit message"]-public
set -euo pipefail
src="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:?usage: make_public.sh <out dir> [commit message]}"
message="${2:-Sync from the development repository}"
existing=0
if [ -e "$out/.git" ]; then
  existing=1
  git -C "$out" diff --quiet && git -C "$out" diff --cached --quiet || { echo "$out has uncommitted changes" >&2; exit 1; }
  git -C "$out" ls-files -z | (cd "$out" && xargs -0 rm -f)  # drop published files; .git stays
fi
mkdir -p "$out"
cd "$src"
files=(
  README.md LICENSE NOTICE CITATION.cff MODEL_CARD.md DESIGN.md pyproject.toml uv.lock .python-version .gitignore
  src tests docs
  configs/gpu_teacher.yaml configs/gpu_decoder.yaml configs/ft/ca179eb54e4e.yaml configs/distill_v2c.yaml
  configs/mac_smoke.yaml configs/mac_decoder_smoke.yaml
  scripts/cloud_bootstrap.sh scripts/fetch_cv.sh scripts/train_forever.sh scripts/bench_codec.py
  scripts/compare_all.sh scripts/compare_systems.py scripts/render_third_party.py scripts/render_freya.py scripts/utmos_score.py
  scripts/web_fixtures.py scripts/web_readings.py scripts/figures.py scripts/make_public.sh
  frontend-rs/Cargo.toml frontend-rs/Cargo.lock frontend-rs/src
  web/package.json web/package-lock.json web/tsconfig.json web/vite.config.ts web/cloudflare.config.ts web/index.html web/src web/test
  web/scripts web/public/_headers web/public/favicon.svg web/public/404.html web/public/fonts
  demo/hard_sentences.json
)
for f in "${files[@]}"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
  if [ -d "$f" ]; then
    mkdir -p "$out/$f" && cp -R "$f/." "$out/$f/"
  else
    mkdir -p "$out/$(dirname "$f")" && cp "$f" "$out/$f"
  fi
done
find "$out" -name __pycache__ -type d -prune -exec rm -rf {} +
cat >> "$out/.gitignore" <<'IGNORE'
frontend-rs/target/
web/node_modules/
web/dist/
web/public/models/
web/public/demo/
web/public/models.json
web/src/generated.ts
web/.wrangler/
web/.cloudflare/
release/
IGNORE
if grep -rIlE "hf_[A-Za-z0-9]{30,}|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}:[0-9]+|ssh -p [0-9]+ root@" "$out"; then
  echo "possible secret or host address above; not committing" >&2
  exit 1
fi
if [ "$existing" = 0 ]; then
  git -C "$out" init -q -b main
  git -C "$out" add -A
  git -C "$out" -c commit.gpgsign=false commit -q -m "Okur v1.0: open Turkish TTS (training, evaluation, PyTorch/MLX/ONNX/WebAssembly runtimes)"
  git -C "$out" tag -a v1.0 -m "Okur v1.0"
else
  find "$out" -type d -empty -not -path "*/.git/*" -delete
  git -C "$out" add -A
  git -C "$out" diff --cached --quiet && { echo "public repo unchanged"; exit 0; }
  git -C "$out" -c commit.gpgsign=false commit -q -m "$message"
fi
echo "public repo ready at $out ($(git -C "$out" ls-files | wc -l | tr -d ' ') files)"
