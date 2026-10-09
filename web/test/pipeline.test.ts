// The TypeScript runtime must reproduce the Python reference (okur.export_web.OnnxSynthesizer) step by step.
import { readFileSync } from "node:fs";
import { join } from "node:path";

import * as ort from "onnxruntime-node";
import { beforeAll, describe, expect, it } from "vitest";

import { chunk } from "../src/core/chunks";
import { Frontend } from "../src/core/frontend";
import { gaussianNoise } from "../src/core/noise";
import { frameTimeline, wordFrames, words } from "../src/core/timeline";
import { type ModelConfig, Tts } from "../src/core/tts";

interface Case {
  text: string;
  letters: string;
  dur: number[];
  cw: number[];
  wstart: number[];
  fw: number[];
  fp: number[];
  noise_head: number[];
  audio_length: number;
  audio_rms: number;
  audio_head: number[];
}

const models = join(import.meta.dirname, "../public/models");
const { cases } = JSON.parse(readFileSync(join(import.meta.dirname, "fixtures.json"), "utf8")) as { cases: Case[] };
const config = JSON.parse(readFileSync(join(import.meta.dirname, "../public/models.json"), "utf8")) as ModelConfig;
let tts: Tts;

beforeAll(async () => {
  const frontend = await Frontend.load(readFileSync(join(models, config.files.frontend!)));
  const open = (name: string) => ort.InferenceSession.create(join(models, name));
  const [text, sample, decoder] = await Promise.all([open(config.files.text), open(config.files.sample),
    open(config.files.decoder)]);
  tts = new Tts(ort, { text, sample, decoder }, frontend, config);
});

describe.each(cases)("$text", (c) => {
  it("reads the text like Python", () => {
    expect(tts.letters(c.text)).toBe(c.letters);
  });

  it("builds the same timeline", () => {
    const w = words(c.letters);
    expect(Array.from(w.cw)).toEqual(c.cw);
    expect(Array.from(w.wstart)).toEqual(c.wstart);
    const { fw, fp } = frameTimeline(wordFrames(Float32Array.from(c.dur), w.cw, w.nWords));
    expect(Array.from(fw)).toEqual(c.fw);
    expect(Array.from(fp)).toEqual(c.fp.map(Math.fround));
  });

  it("draws the same noise", () => {
    const noise = gaussianNoise(0, c.fw.length * config.latent_dim * config.times.length);
    Array.from(noise.subarray(0, 64)).forEach((x, i) => expect(x).toBeCloseTo(c.noise_head[i]!, 5));
  });

  it("produces the same audio", async () => {
    expect(chunk(c.letters)).toEqual([c.letters]);
    const out = await tts.chunk(c.letters, 1, 0);
    expect(out.audio.length).toBe(c.audio_length);
    const rms = Math.sqrt(out.audio.reduce((a, x) => a + x * x, 0) / out.audio.length);
    expect(Math.abs(rms - c.audio_rms) / c.audio_rms).toBeLessThan(1e-3);
    const worst = c.audio_head.reduce((m, x, i) => Math.max(m, Math.abs(x - out.audio[i]!)), 0);
    expect(worst).toBeLessThan(1e-3);
  });
});

describe("chunking", () => {
  it("keeps sentences whole and short pieces together", () => {
    expect(chunk("bir. iki üç dört beş altı yedi sekiz dokuz on on bir on iki on üç on dört on beş. üç!"))
      .toEqual(["bir. iki üç dört beş altı yedi sekiz dokuz on on bir on iki on üç on dört on beş.", "üç!"]);
  });

  it("splits run-on text at spaces under the limit", () => {
    const pieces = chunk(Array.from({ length: 200 }, () => "kelime").join(" "), 100);
    expect(pieces.every((p) => p.length <= 100 && !p.startsWith(" "))).toBe(true);
    expect(pieces.join(" ")).toBe(Array.from({ length: 200 }, () => "kelime").join(" "));
  });
});
