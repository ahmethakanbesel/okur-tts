/**
 * Text to speech with the exported graphs (okur.export_web), independent of the ONNX Runtime flavour: the browser
 * passes onnxruntime-web, tests pass onnxruntime-node. Mirrors okur.export_web.OnnxSynthesizer.
 */
import type { InferenceSession, Tensor, TypedTensor } from "onnxruntime-common";

import { chunk } from "./chunks";
import type { Frontend } from "./frontend";
import { gaussianNoise } from "./noise";
import { frameTimeline, wordFrames, words } from "./timeline";

export interface ModelConfig {
  voice: string;
  sample_rate: number;
  hop: number;
  latent_dim: number;
  times: number[];
  symbols: string[];
  files: { text: string; sample: string; decoder: string; frontend?: string };
  bytes: Record<string, number>;
}

/** The parts of onnxruntime (web or node) that this module uses. */
export interface Runtime {
  Tensor: typeof Tensor;
}

export interface Sessions {
  text: InferenceSession;
  sample: InferenceSession;
  decoder: InferenceSession;
}

export interface Chunk {
  letters: string;
  /** Frames (hop / sample_rate seconds each) per letter, after the speed setting: when each letter is spoken. */
  dur: Float32Array<ArrayBuffer>;
  audio: Float32Array<ArrayBuffer>;
  seconds: number;
  computeMs: number;
}

export interface SpeakOptions {
  speed?: number;
  seed?: number;
  /** Multiplies the durations of spaces and punctuation only: longer or shorter breaks at the same speaking rate. */
  pause?: number;
  /** Scales the sampling noise ("expressiveness"); the student was trained at 1. */
  temperature?: number;
}

const PAUSE_SYMBOLS = new Set([" ", ",", ".", ";", ":", "!", "?", "\"", "(", ")"]);

const scaled = (x: Float32Array, k: number): Float32Array => (k === 1 ? x : x.map((v) => v * k));
const int64 = (a: Int32Array | number[]): BigInt64Array => BigInt64Array.from(a, (x) => BigInt(x));

export class Tts {
  private readonly ids: Map<string, number>;

  constructor(
    private readonly ort: Runtime,
    private readonly sessions: Sessions,
    readonly frontend: Frontend,
    readonly config: ModelConfig,
  ) {
    this.ids = new Map(config.symbols.map((s, i) => [s, i]));
  }

  letters(text: string): string {
    return this.frontend.letters(text);
  }

  /** Synthesize chunk by chunk, so playback can start after the first sentence. */
  async *speak(text: string, options: SpeakOptions = {}): AsyncGenerator<Chunk> {
    const letters = this.letters(text);
    let seed = options.seed ?? 0;
    for (const piece of chunk(letters)) {
      yield await this.chunk(piece, options.speed ?? 1, seed, options.pause ?? 1, options.temperature ?? 1);
      seed += 1;
    }
  }

  async chunk(letters: string, speed: number, seed: number, pause = 1, temperature = 1): Promise<Chunk> {
    const started = performance.now();
    const { Tensor } = this.ort;
    const chars = Array.from(letters);
    const n = chars.length;
    const ids = new Tensor("int64", int64(chars.map((c) => this.ids.get(c) ?? 1)), [1, n]);
    const text = await this.sessions.text.run({ ids });
    const dur = Float32Array.from((text.dur as TypedTensor<"float32">).data,
      (d, i) => (d / speed) * (pause !== 1 && PAUSE_SYMBOLS.has(chars[i] ?? "") ? pause : 1));
    const w = words(letters);
    const { fw, fp } = frameTimeline(wordFrames(dur, w.cw, w.nWords));
    const t = fw.length;
    const steps = this.config.times.length;
    const c = this.config.latent_dim;
    const sampled = await this.sessions.sample.run({
      h: text.h as Tensor,
      dur: new Tensor("float32", dur, [1, n]),
      cw: new Tensor("int64", int64(w.cw), [1, n]),
      wstart: new Tensor("int64", int64(w.wstart), [1, n]),
      fw: new Tensor("int64", int64(fw), [1, t]),
      fp: new Tensor("float32", fp, [1, t]),
      noise: new Tensor("float32", scaled(gaussianNoise(seed, steps * t * c), temperature), [steps, 1, t, c]),
    });
    const decoded = await this.sessions.decoder.run({ z: sampled.z as Tensor });
    const audio = Float32Array.from((decoded.audio as TypedTensor<"float32">).data);
    return { letters, dur, audio, seconds: audio.length / this.config.sample_rate, computeMs: performance.now() - started };
  }
}
