/// Runs the whole pipeline off the main thread: model download, the WebAssembly frontend and ONNX Runtime.
import * as ort from "onnxruntime-web/wasm";

import { chunk } from "./core/chunks";
import { Frontend } from "./core/frontend";
import { type ModelConfig, Tts } from "./core/tts";
import { ORT_PATH } from "./generated";
import type { Reply, Request } from "./protocol";

const port = self as unknown as Worker;
const post = (message: Reply, transfer: Transferable[] = []): void => port.postMessage(message, transfer);
const MODELS = "/models/";

let ready: Promise<Tts> | undefined;
let config: Promise<ModelConfig> | undefined;
let frontendReady: Promise<Frontend> | undefined;
let frontendBytes = 0;

const loadConfig = (): Promise<ModelConfig> =>
  (config ??= fetch("/models.json", { cache: "no-cache" }).then((r) => {
    if (!r.ok) throw new Error(`/models.json: HTTP ${r.status}`);
    return r.json() as Promise<ModelConfig>;
  }));

/** The 1.3 MB text frontend loads first, so the page can show the reading before the 24 MB model arrives. */
const loadFrontend = (onBytes: (n: number) => void = () => {}): Promise<Frontend> =>
  (frontendReady ??= loadConfig().then((c) => download(MODELS + c.files.frontend!, (n) => {
    frontendBytes += n;
    onBytes(n);
  })).then((b) => Frontend.load(b)));

async function download(url: string, onBytes: (n: number) => void): Promise<Uint8Array<ArrayBuffer>> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`${url}: HTTP ${response.status}`);
  const parts: Uint8Array<ArrayBuffer>[] = [];
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    onBytes(value.length);
  }
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

async function load(requested?: number): Promise<Tts> {
  const started = performance.now();
  const config = await loadConfig();
  const files = config.files;
  const total = Object.values(config.bytes).reduce((a, b) => a + b, 0);
  let loaded = frontendBytes;
  const tick = (n: number): void => {
    loaded += n;
    post({ type: "progress", loaded, total });
  };
  const threads = requested ?? (self.crossOriginIsolated ? Math.max(1, Math.min(8, navigator.hardwareConcurrency - 1)) : 1);
  ort.env.wasm.wasmPaths = ORT_PATH;
  ort.env.wasm.numThreads = threads;
  const options: ort.InferenceSession.SessionOptions = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const [frontend, text, sample, decoder] = await Promise.all([
    loadFrontend(tick),
    download(MODELS + files.text, tick).then((b) => ort.InferenceSession.create(b, options)),
    download(MODELS + files.sample, tick).then((b) => ort.InferenceSession.create(b, options)),
    download(MODELS + files.decoder, tick).then((b) => ort.InferenceSession.create(b, options)),
  ]);
  post({ type: "stage", message: "Preparing the model…" });
  const tts = new Tts(ort, { text, sample, decoder }, frontend, config);
  await tts.chunk("merhaba.", 1, 0); // warm up kernels so the first real request is fast
  post({ type: "ready", threads, isolated: self.crossOriginIsolated, loadMs: performance.now() - started });
  return tts;
}

let latest = 0; // a newer request cancels older ones between chunks

port.onmessage = async (event: MessageEvent<Request>) => {
  const request = event.data;
  try {
    if (request.type === "cancel") {
      latest = 0;
      return;
    }
    if (request.type === "load") {
      ready ??= load(request.threads);
      await ready;
      return;
    }
    if (request.type === "letters") {
      post({ type: "letters", id: request.id, letters: (await loadFrontend()).letters(request.text) });
      return;
    }
    ready ??= load();
    const tts = await ready;
    latest = request.id;
    const letters = tts.letters(request.text);
    post({ type: "letters", id: request.id, letters });
    const pieces = chunk(letters);
    let seed = request.seed;
    for (const [index, piece] of pieces.entries()) {
      if (latest !== request.id) return;
      const out = await tts.chunk(piece, request.speed, seed++, request.pause, request.temperature);
      post({ type: "chunk", id: request.id, index, count: pieces.length, letters: out.letters, dur: out.dur,
        audio: out.audio, seconds: out.seconds, computeMs: out.computeMs }, [out.audio.buffer, out.dur.buffer]);
    }
    post({ type: "done", id: request.id });
  } catch (error) {
    const id = "id" in request ? request.id : undefined;
    post({ type: "error", ...(id === undefined ? {} : { id }), message: error instanceof Error ? error.message : String(error) });
  }
};
