# Training okur from scratch

The released model was trained on one rented RTX 5090 (Vast.ai, $0.56/h) in about 21 GPU hours, about $12 in total. Every stage
resumes from its last checkpoint after an interruption, and a killed-and-resumed run is bitwise identical to an
uninterrupted one.

## 0. Set up

```sh
uv sync                      # Python 3.13; groups: dev, data, mlx (Apple only), export, demo
uv run pytest -q             # frontend, model, training resume/rollback, MLX/ONNX parity, Rust frontend parity
```

On a fresh GPU machine, `scripts/cloud_bootstrap.sh` installs everything, downloads the open corpora and prepares
them (≈3 h on one GPU). Common Voice needs a Mozilla Data Collective API key (`MDC_API_KEY`, terms accepted on the
dataset page); the other corpora come from the Hugging Face Hub (`HF_TOKEN` for rate limits).

## 1. Data: 283,871 clips, 298 h, 44 speakers

| Source | Kept | License |
|---|---|---|
| ISSAI Turkish Speech Corpus | 204 h | MIT |
| Mozilla Common Voice 27.0, Turkish (speakers with ≥ 300 clips keep their identity; the rest are pooled) | 75 h | CC0 |
| MediaSpeech, Turkish | 9 h | CC BY 4.0 |
| Google FLEURS, Turkish | 6 h | CC BY 4.0 |

`okur prepare <source> <raw dir> <out dir>` writes Parquet shards. Per clip it:

1. Normalizes the transcript with the same frontend used at inference (`okur.frontend`).
2. Force-aligns letters with a Turkish wav2vec2 CTC model (`mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`) and its own
   Viterbi pass, giving letter durations in 40 ms frames and an alignment score used for filtering.
3. Encodes 48 kHz audio into 64-dim, 25 Hz latents with the frozen ACE-Step 1.5 VAE (TF32; batches sized by audio
   seconds, not clip count).
4. Tags the original bandwidth (16/22/48 kHz) as a quality bucket, so inference can ask for clean full-band audio.
5. Stores text, letters, durations, latents (fp16) and the cropped audio (FLAC, for the decoder GAN).

Freya-TR-Eval sentences are evaluation-only; the evaluation reports a "clean" score that excludes the 89 sentences
that also occur as Common Voice prompts.

## 2. Acoustic teacher (multi-speaker, 16 steps)

```sh
KIND=acoustic scripts/train_forever.sh configs/gpu_teacher.yaml   # 120k steps, ~1 h 50 min
```

Flow matching on normalized latents (logit-normal timesteps, 10% condition dropout for classifier-free guidance) plus
log-duration regression. Eager mode is used on purpose: with variable batch shapes, `torch.compile` and CUDA graphs
were 16× slower. `train_forever.sh` is a supervisor: it restarts on crashes, watches a heartbeat, and stops cleanly
on `runs/HOLD`.

## 3. Decoder (3 M parameters, latents → 48 kHz)

```sh
KIND=decoder scripts/train_forever.sh configs/gpu_decoder.yaml    # 110k steps at batch 16, ~11 h
```

HiFi-GAN-style generator (upsampling 8·6·5·2·2·2 = 1920) trained with multi-period and multi-scale discriminators,
feature matching and a multi-resolution mel loss, on real audio paired with its ACE-Step latents. Intelligibility
stopped improving after ~70k steps; naturalness (UTMOS) kept improving to 110k.

## 4. One voice

```sh
uv run okur train configs/ft/ca179eb54e4e.yaml                # 3k steps from the teacher, ~2 min
```

Four candidate voices (Common Voice contributors with the most clean hours) were fine-tuned and compared; the one with
the lowest recognizer error was kept.

## 5. DMD2 distillation to 4 steps

```sh
KIND=distill scripts/train_forever.sh configs/distill_v2c.yaml    # step 6k is the release
```

Distribution-matching distillation with a frozen teacher (real score, guidance 3 baked in) and a trainable fake-score
model updated five times per student step. A gentle setting matters: student lr 5e-6 and fake lr 2e-5. At a 4× higher
student lr the student drifted after ~3k steps.

## 6. Evaluate and package

```sh
uv run okur eval-freya <student> data/prepared/*/ --decoder <decoder> --speaker cv:ca179eb54e4e
scripts/compare_all.sh                                            # demo comparison: UTMOS, WER, speed, samples
uv run okur release <student> <decoder> cv:ca179eb54e4e --out release   # safetensors + ONNX
uv run okur say "Merhaba dünya." --model release
```

## Lessons

- The codec must batch by audio seconds, or long clips run out of GPU memory.
- The teacher and the decoder GAN don't fit on one 32 GB GPU together; run them one after the other.
- Early stopping on a 6-sentence recognizer CER is too noisy for the teacher; train for a fixed number of steps.
- DMD2 is sensitive: evaluate every 1–3k student steps with the full Freya set, not the training-time CER.
- Most of the remaining gap to EMA Lightning comes from the data (≈5 h of one crowd-sourced voice versus about
  1,000 h of one studio voice), not from the architecture.
