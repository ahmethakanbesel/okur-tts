import "./style.css";
import "./keycaps.css";

import { asciiName, type ExportFormat, exportAudio, extension } from "./core/export";
import { CATEGORIES, EXAMPLES, type Key, type Lang, apply, initialLang, number, percent, rememberLang, t } from "./i18n";
import type { Reply, Request } from "./protocol";
import readings from "./readings.json";
import ditEn from "./figures/dit.en.svg?raw";
import ditTr from "./figures/dit.tr.svg?raw";
import pipelineEn from "./figures/pipeline.en.svg?raw";
import pipelineTr from "./figures/pipeline.tr.svg?raw";

const SAMPLE_RATE = 48000;

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el as T;
};

const ui = {
  lang: $<HTMLButtonElement>("lang"),
  text: $<HTMLTextAreaElement>("text"),
  speak: $<HTMLButtonElement>("speak"),
  speakLabel: $("speak-label"),
  vary: $<HTMLButtonElement>("vary"),
  download: $<HTMLAnchorElement>("download"),
  status: $("status"),
  statusText: $("status-text"),
  feed: $("feed"),
  reading: $("reading"),
  examples: $("examples"),
  samples: $("samples"),
  metrics: $<HTMLTableElement>("metrics"),
  links: $("links"),
  more: $<HTMLButtonElement>("more"),
};
const FIRST_SAMPLES = 6;
let showAll = false;

// ---- language ------------------------------------------------------------------------------------------------

let lang: Lang = initialLang();
let lastStatus: { key: Key; vars: Record<string, string | number>; error: boolean } = { key: "status.first", vars: {},
  error: false };

function setStatus(key: Key, vars: Record<string, string | number> = {}, error = false): void {
  lastStatus = { key, vars, error };
  ui.statusText.textContent = t(lang, key, vars);
  ui.status.classList.toggle("error", error);
}

const FIGURES: Record<string, Record<Lang, string>> = {
  "fig-pipeline": { tr: pipelineTr, en: pipelineEn },
  "fig-dit": { tr: ditTr, en: ditEn },
};

function renderFigures(): void {
  for (const [id, svg] of Object.entries(FIGURES)) {
    const host = document.getElementById(id);
    // each figure gets its own arrowhead id; our own build-time SVG, never user input
    if (host) host.innerHTML = svg[lang].replaceAll('id="ah"', `id="ah-${id}"`).replaceAll("url(#ah)", `url(#ah-${id})`);
  }
}

function setLang(next: Lang): void {
  lang = next;
  apply(lang);
  renderFigures();
  showOptions();
  ui.lang.setAttribute("aria-checked", String(lang === "en"));
  setStatus(lastStatus.key, lastStatus.vars, lastStatus.error);
  setSpeaking(Boolean(ui.speak.dataset.speaking));
  renderExamples();
  renderSamples();
  renderMetrics();
}

ui.lang.addEventListener("click", () => {
  const next = lang === "tr" ? "en" : "tr";
  rememberLang(next);
  const url = new URL(location.href);
  url.searchParams.set("lang", next);
  history.replaceState(null, "", url);
  setLang(next);
});

// ---- the synthesis worker --------------------------------------------------------------------------------------

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
worker.onerror = (e) => setStatus("status.worker", { m: e.message || "?" }, true);
const send = (request: Request): void => worker.postMessage(request);

// ---- options (advanced drawer); remembered per browser as a convenience ---------------------------------------

interface Options {
  speed: number;
  pause: number;
  gap: number;
  temperature: number;
  seed: number;
  format: ExportFormat;
  dictionary: string;
}

const DEFAULTS: Options = { speed: 1, pause: 1, gap: 0, temperature: 1, seed: 0, format: "wav", dictionary: "" };
const FORMATS: ExportFormat[] = ["wav", "mp3-48", "mp3-24", "mp3-16"];

function loadOptions(): Options {
  try {
    const saved = JSON.parse(localStorage.getItem("okur.options") ?? "{}") as Partial<Options>;
    const o = { ...DEFAULTS, ...saved };
    return FORMATS.includes(o.format) ? o : { ...o, format: "wav" };
  } catch {
    return { ...DEFAULTS };
  }
}

let opts = loadOptions();

function saveOptions(): void {
  try {
    localStorage.setItem("okur.options", JSON.stringify(opts));
  } catch {
    // not remembered; the settings still apply for this visit
  }
}

/** User pronunciation rules ("written = spoken", one per line), applied to whole words before the frontend. */
function pronounce(text: string): string {
  let out = text;
  for (const line of opts.dictionary.split("\n")) {
    const at = line.indexOf("=");
    if (at <= 0) continue;
    const written = line.slice(0, at).trim();
    const spoken = line.slice(at + 1).trim();
    if (!written || !spoken) continue;
    const escaped = written.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    out = out.replace(new RegExp(`(?<![\\p{L}\\p{N}])${escaped}(?![\\p{L}\\p{N}])`, "giu"), spoken);
  }
  return out;
}
let nextId = 1;
let current = 0;
let modelReady = false;
let collected: Float32Array<ArrayBuffer>[] = [];
let produced = { seconds: 0, ms: 0 };

let audioContext: AudioContext | undefined;
let playhead = 0;
let liveSources: AudioBufferSourceNode[] = [];

const context = (): AudioContext => (audioContext ??= new AudioContext({ sampleRate: SAMPLE_RATE }));

function stopLive(): void {
  for (const s of liveSources) s.stop();
  liveSources = [];
  playhead = 0;
  strikes = [];
}

/** Schedule a chunk of audio after `gap` seconds of silence; returns its start time on the audio clock. */
function schedule(samples: Float32Array<ArrayBuffer>, gap = 0): number {
  const ctx = context();
  const buffer = ctx.createBuffer(1, samples.length, SAMPLE_RATE);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  playhead = Math.max(playhead, ctx.currentTime + 0.05) + gap;
  const start = playhead;
  source.start(start);
  playhead += buffer.duration;
  liveSources.push(source);
  source.onended = () => {
    liveSources = liveSources.filter((s) => s !== source);
    if (liveSources.length === 0 && current === 0) setSpeaking(false);
  };
  return start;
}

function setSpeaking(on: boolean): void {
  ui.speakLabel.textContent = t(lang, on ? "key.stop" : "key.read");
  if (on) ui.speak.dataset.speaking = "1";
  else delete ui.speak.dataset.speaking;
}

// ---- the reading line: typed in time with the voice -------------------------------------------------------------

const turkishLower = (s: string): string => s.replaceAll("İ", "i").replaceAll("I", "ı").toLocaleLowerCase("tr");
const plain = (s: string): string => s.replace(/[âîû]/g, (c) => ({ â: "a", î: "i", û: "u" })[c] ?? c);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

let chars: HTMLSpanElement[] = [];
let readingText = "";
let cursor = 0;
let strikes: { at: number; index: number }[] = [];
let raf = 0;

/** Lay out the letters the model reads, every character unstruck; words the frontend changed get the red ribbon,
 *  and restored circumflexes keep a ghost of what was typed underneath. */
function layReading(input: string, letters: string): void {
  const typed = new Set(turkishLower(input).split(/[^\p{L}]+/u).filter(Boolean));
  chars = [];
  readingText = letters;
  cursor = 0;
  const parts: Node[] = [];
  // One span per space-separated token, so punctuation stays with its word when lines wrap.
  for (const piece of letters.split(/( +)/u)) {
    if (!piece) continue;
    const holder = document.createElement("span");
    const word = piece.match(/\p{L}+/u)?.[0];
    if (word) {
      holder.className = "w";
      if (!typed.has(word)) {
        holder.classList.add("changed");
        const was = plain(word);
        if (was !== word && typed.has(was) && piece.startsWith(word)) holder.dataset.was = was;
      }
    }
    for (const c of piece) {
      const ch = document.createElement("span");
      ch.className = "ch";
      ch.textContent = c;
      holder.append(ch);
      chars.push(ch);
    }
    parts.push(holder);
  }
  ui.reading.replaceChildren(...parts);
}

/** Every letter of `piece` strikes when its sound begins: onsets from the model's own durations, scaled to the audio. */
function timeStrikes(piece: string, dur: Float32Array, start: number, seconds: number): void {
  const offset = readingText.indexOf(piece, cursor);
  if (offset < 0) return;
  cursor = offset + piece.length;
  const total = dur.reduce((a, b) => a + b, 0) || 1;
  let elapsed = 0;
  for (let i = 0; i < piece.length; i++) {
    strikes.push({ at: start + (elapsed / total) * seconds, index: offset + i });
    elapsed += dur[i] ?? 0;
  }
  if (!raf) raf = requestAnimationFrame(strikeLoop);
}

function strikeLoop(): void {
  raf = 0;
  // Audio the browser refuses to start (no user gesture, autoplay policy): show the whole reading at once.
  const now = audioContext?.state === "running" ? audioContext.currentTime : Number.POSITIVE_INFINITY;
  const due = strikes.filter((s) => s.at <= now);
  strikes = strikes.filter((s) => s.at > now);
  for (const s of due) {
    const ch = chars[s.index];
    if (!ch) continue;
    ch.classList.add("on");
    if (!reduceMotion.matches) ch.classList.add("strike");
  }
  if (strikes.length) raf = requestAnimationFrame(strikeLoop);
}

// ---- speaking -------------------------------------------------------------------------------------------------

function speak(): void {
  if (ui.speak.dataset.speaking) {
    current = 0;
    send({ type: "cancel" });
    stopLive();
    for (const ch of chars) ch.classList.add("on");
    setSpeaking(false);
    return;
  }
  const text = pronounce(ui.text.value.trim());
  if (!text || !modelReady) return;
  lastText = ui.text.value.trim();
  stopPlayer();
  stopLive();
  void context().resume(); // inside the click, for Safari
  current = nextId++;
  collected = [];
  produced = { seconds: 0, ms: 0 };
  ui.download.hidden = true;
  setSpeaking(true);
  setStatus("status.reading");
  if (matchMedia("(max-width: 760px)").matches) { // the docked panel must not cover the line being typed
    document.querySelector(".reading-label")?.scrollIntoView({ block: "start", behavior: reduceMotion.matches ? "auto" : "smooth" });
  }
  send({ type: "speak", id: current, text, speed: opts.speed, seed: opts.seed, pause: opts.pause,
    temperature: opts.temperature });
}

// The reading at rest: precomputed for the default and example texts, asked of the WebAssembly frontend for anything
// typed (it loads before the model). Every letter waits in ghost ink; OKU strikes them forward with the voice.
const known = readings as Record<string, string>;
let previewId = -1;
let previewTimer = 0;

function preview(): void {
  if (ui.speak.dataset.speaking) return;
  const text = pronounce(ui.text.value.trim());
  const letters = known[text];
  if (letters !== undefined) {
    layReading(ui.text.value, letters);
    return;
  }
  if (!text) return;
  previewId -= 1;
  send({ type: "letters", id: previewId, text });
}

ui.text.addEventListener("input", () => {
  autosize();
  clearTimeout(previewTimer);
  previewTimer = window.setTimeout(preview, 250);
});

/** Grow the textarea with its text where CSS field-sizing is unsupported (Firefox, older Safari). */
function autosize(): void {
  if (CSS.supports("field-sizing", "content")) return;
  ui.text.style.height = "auto";
  ui.text.style.height = `${Math.min(ui.text.scrollHeight + 2, 12 * 1.6 * parseFloat(getComputedStyle(ui.text).fontSize))}px`;
}

function restart(): void {
  if (ui.speak.dataset.speaking) speak();
  speak();
}

worker.onmessage = (event: MessageEvent<Reply>) => {
  const m = event.data;
  switch (m.type) {
    case "progress":
      ui.feed.style.setProperty("--p", String(Math.min(1, m.loaded / m.total)));
      setStatus("status.loading", { a: number(lang, m.loaded / 1e6, 1), b: number(lang, m.total / 1e6, 1) });
      break;
    case "stage":
      setStatus("status.preparing");
      break;
    case "ready":
      modelReady = true;
      ui.feed.dataset.done = "1";
      ui.speak.disabled = false;
      ui.vary.disabled = false;
      setStatus("status.ready", { n: m.threads, s: number(lang, m.loadMs / 1000, 1) });
      break;
    case "letters":
      if (m.id === current || (m.id === previewId && !ui.speak.dataset.speaking)) layReading(ui.text.value, m.letters);
      break;
    case "chunk": {
      if (m.id !== current) break;
      const gap = m.index > 0 ? opts.gap : 0;
      const start = schedule(m.audio, gap);
      timeStrikes(m.letters, m.dur, start, m.seconds);
      if (gap > 0) collected.push(new Float32Array(Math.round(gap * SAMPLE_RATE)));
      collected.push(m.audio);
      produced.seconds += m.seconds;
      produced.ms += m.computeMs;
      const vars = { a: number(lang, produced.seconds, 1), b: number(lang, produced.ms / 1000, 2),
        r: number(lang, produced.seconds / (produced.ms / 1000), 1) };
      setStatus("status.result", vars);
      if (m.count > 1) ui.statusText.textContent += t(lang, "status.part", { i: m.index + 1, n: m.count });
      break;
    }
    case "done": {
      if (m.id !== current) break;
      current = 0;
      const all = new Float32Array(collected.reduce((n, a) => n + a.length, 0));
      let offset = 0;
      for (const a of collected) {
        all.set(a, offset);
        offset += a.length;
      }
      lastAudio = all;
      ui.download.hidden = false;
      if (liveSources.length === 0) setSpeaking(false);
      break;
    }
    case "error":
      if (m.id === undefined || m.id === current) {
        current = 0;
        setSpeaking(false);
        ui.feed.dataset.done = "1";
        setStatus("status.error", { m: m.message }, true);
      }
      break;
  }
};

ui.speak.addEventListener("click", speak);
ui.vary.addEventListener("click", () => {
  opts.seed = (opts.seed + 1) % 1000000;
  saveOptions();
  showOptions();
  restart();
});

// Save the last reading in the chosen format, named after its text.
let lastAudio: Float32Array<ArrayBuffer> | undefined;
let lastText = "";
ui.download.href = "#";
ui.download.addEventListener("click", (e) => {
  e.preventDefault();
  if (!lastAudio) return;
  const audio = lastAudio;
  const format = opts.format;
  void exportAudio(audio, SAMPLE_RATE, format).then((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${asciiName(lastText)}_okur.${extension(format)}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }, (error: unknown) => setStatus("status.error", { m: error instanceof Error ? error.message : String(error) }, true));
});
ui.text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && modelReady) {
    e.preventDefault();
    restart();
  }
});
for (const button of document.querySelectorAll<HTMLButtonElement>(".speed button")) {
  button.addEventListener("click", () => {
    opts.speed = Number(button.dataset.speed);
    saveOptions();
    showOptions();
  });
}

// ---- the advanced drawer ---------------------------------------------------------------------------------------

const field = <T extends HTMLElement>(id: string): T => $<T>(id);
const controls = {
  speed: field<HTMLInputElement>("opt-speed"), pause: field<HTMLInputElement>("opt-pause"),
  gap: field<HTMLInputElement>("opt-gap"), temperature: field<HTMLInputElement>("opt-temp"),
  seed: field<HTMLInputElement>("opt-seed"), format: field<HTMLSelectElement>("opt-format"),
  dictionary: field<HTMLTextAreaElement>("opt-dict"),
};
const outputs = { speed: $("out-speed"), pause: $("out-pause"), gap: $("out-gap"), temperature: $("out-temp") };

/** Reflect `opts` in every control, the speed presets and the download key. */
function showOptions(): void {
  controls.speed.value = String(opts.speed);
  controls.pause.value = String(opts.pause);
  controls.gap.value = String(opts.gap);
  controls.temperature.value = String(opts.temperature);
  controls.seed.value = String(opts.seed);
  controls.format.value = opts.format;
  if (document.activeElement !== controls.dictionary) controls.dictionary.value = opts.dictionary;
  outputs.speed.textContent = `${number(lang, opts.speed, 2)}×`;
  outputs.pause.textContent = `${number(lang, opts.pause, 1)}×`;
  outputs.gap.textContent = t(lang, "adv.seconds", { v: number(lang, opts.gap, 2) });
  outputs.temperature.textContent = number(lang, opts.temperature, 2);
  for (const b of document.querySelectorAll<HTMLButtonElement>(".speed button")) {
    b.setAttribute("aria-checked", String(Math.abs(Number(b.dataset.speed) - opts.speed) < 1e-6));
  }
  $("download-label").textContent = t(lang, opts.format === "wav" ? "key.download.wav" : "key.download.mp3");
}

for (const key of ["speed", "pause", "gap", "temperature"] as const) {
  controls[key].addEventListener("input", () => {
    opts[key] = Number(controls[key].value);
    saveOptions();
    showOptions();
  });
}
controls.seed.addEventListener("change", () => {
  const value = Math.max(0, Math.min(999999, Math.round(Number(controls.seed.value) || 0)));
  opts.seed = value;
  saveOptions();
  showOptions();
});
controls.format.addEventListener("change", () => {
  opts.format = controls.format.value as ExportFormat;
  saveOptions();
  showOptions();
});
controls.dictionary.addEventListener("input", () => {
  opts.dictionary = controls.dictionary.value;
  saveOptions();
  clearTimeout(previewTimer);
  previewTimer = window.setTimeout(preview, 300);
});
$("opt-reset").addEventListener("click", () => {
  opts = { ...DEFAULTS };
  saveOptions();
  showOptions();
  preview();
});

function useText(text: string): void {
  ui.text.value = text;
  autosize();
  if (modelReady) restart();
  else preview();
}

function renderExamples(): void {
  const label = ui.examples.querySelector(".label-inline");
  const parts: Node[] = label ? [label] : [];
  EXAMPLES.forEach((example, i) => {
    if (i > 0) {
      const sep = document.createElement("span");
      sep.className = "sep";
      sep.textContent = "/";
      sep.setAttribute("aria-hidden", "true");
      parts.push(sep);
    }
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = example[lang];
    b.addEventListener("click", () => useText(example.text));
    parts.push(b);
  });
  ui.examples.replaceChildren(...parts);
}

// ---- hard sentences and numbers -----------------------------------------------------------------------------

interface Sample {
  id: string;
  text: string;
  audio: Record<string, string>;
}

interface SystemMetrics {
  label: string;
  reported?: boolean; // published by its authors under another protocol: shown apart, never ranked
  model_url?: string;
  license?: string;
  params?: string;
  rtf?: number;
  utmos?: number;
  utmos_ci95?: number;
  wer_all?: number;
  wer_clean?: number;
  cer_clean?: number;
}

interface Metrics {
  systems: Record<string, SystemMetrics>;
  links?: { label: string; url: string }[];
}

let samples: Sample[] = [];
let metrics: Metrics | undefined;

const player = new Audio();
player.preload = "none";
let playing: HTMLButtonElement | undefined;

function stopPlayer(): void {
  player.pause();
  playing?.setAttribute("aria-pressed", "false");
  playing = undefined;
}

player.addEventListener("ended", stopPlayer);

function play(button: HTMLButtonElement, src: string): void {
  const again = playing === button;
  stopPlayer();
  if (ui.speak.dataset.speaking) speak();
  if (again) return;
  player.src = src;
  void player.play();
  playing = button;
  button.setAttribute("aria-pressed", "true");
}

const svgKey = (): SVGSVGElement => {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 12 12");
  svg.setAttribute("aria-hidden", "true");
  const tri = document.createElementNS(ns, "path");
  tri.setAttribute("class", "g-play");
  tri.setAttribute("d", "M2 1 L11 6 L2 11 Z");
  const sq = document.createElementNS(ns, "rect");
  sq.setAttribute("class", "g-stop");
  for (const [k, v] of Object.entries({ x: "2", y: "2", width: "8", height: "8" })) sq.setAttribute(k, v);
  svg.append(tri, sq);
  return svg;
};

/** A drawn download glyph: an arrow onto a tray. */
const svgDownload = (): SVGSVGElement => {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 12 12");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", "M6 1.5 V7.5 M3.5 5 L6 7.5 L8.5 5 M2 9.5 V10.5 H10 V9.5");
  svg.append(path);
  return svg;
};

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] => {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

function renderSamples(): void {
  if (!samples.length) return;
  const systems = metrics?.systems ?? {};
  ui.more.hidden = samples.length <= FIRST_SAMPLES;
  ui.more.textContent = t(lang, showAll ? "hard.less" : "hard.more");
  ui.more.setAttribute("aria-expanded", String(showAll));
  ui.samples.replaceChildren(...samples.slice(0, showAll ? samples.length : FIRST_SAMPLES).map((s) => {
    const li = el("li", "card");
    const tab = el("div", "card-tab");
    tab.append(el("div", "cat", CATEGORIES[lang][s.id] ?? s.id));
    const live = el("button", "keycap live", t(lang, "hard.live"));
    live.type = "button";
    live.addEventListener("click", () => {
      useText(s.text);
      document.getElementById("try")?.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth" });
    });
    tab.append(live);
    const body = el("div", "card-body");
    const sentence = el("p", "sentence", s.text);
    sentence.lang = "tr";
    const row = el("div", "systems");
    for (const [key, src] of Object.entries(s.audio)) {
      const name = systems[key]?.label ?? key;
      const col = el("div", key === "ours" ? "system ours" : "system");
      const b = el("button", "play");
      b.type = "button";
      b.setAttribute("aria-pressed", "false");
      b.setAttribute("aria-label", `${name}: ${s.text}`);
      const cap = el("span", "cap");
      cap.append(svgKey());
      // keep short suffixes attached (MMS‑TTS), let longer names break after the hyphen (FreyaTTS- small)
      const label = el("span", "", name.replace(/-(?=\p{L}{1,3}\b)/gu, "\u2011"));
      label.lang = "en"; // product names: English capitalization (I, not İ) under text-transform
      b.append(cap, label);
      b.addEventListener("click", () => play(b, `/demo/${src}`));
      const dl = el("a", "dl");
      dl.href = `/demo/${src}`;
      dl.download = `${asciiName(s.text)}_${key === "ours" ? "okur" : key}.mp3`;
      dl.title = t(lang, "hard.download");
      dl.setAttribute("aria-label", `${t(lang, "hard.download")}: ${name}, ${s.text}`);
      dl.append(svgDownload());
      const head = el("div", "system-head");
      head.append(b, dl);
      col.append(head);
      row.append(col);
    }
    body.append(sentence, row);
    li.append(tab, body);
    return li;
  }));
}

ui.more.addEventListener("click", () => {
  showAll = !showAll;
  renderSamples();
});

function renderMetrics(): void {
  const all = Object.entries(metrics?.systems ?? {});
  if (!all.length) return;
  const rows = all.filter(([, m]) => !m.reported);
  const reported = all.filter(([, m]) => m.reported);
  const best = (field: keyof SystemMetrics, higher: boolean): number | undefined => {
    const values = rows.map(([, m]) => m[field]).filter((v): v is number => typeof v === "number");
    return values.length ? (higher ? Math.max(...values) : Math.min(...values)) : undefined;
  };
  const top = { utmos: best("utmos", true), wer: best("wer_all", false), werc: best("wer_clean", false),
    cerc: best("cer_clean", false), rtf: best("rtf", true) };
  const cell = (text: string, num = true, isBest = false): HTMLTableCellElement => {
    const td = el("td", `${num ? "num" : ""}${isBest ? " best" : ""}`.trim(), text);
    return td;
  };
  if (!ui.metrics.tHead?.querySelector(".rack")) {
    const rack = el("tr", "rack");
    rack.setAttribute("aria-hidden", "true");
    for (const th of ui.metrics.tHead?.rows[0]?.cells ?? []) {
      const cell = el("td", th.classList.contains("num") ? "stop" : "");
      rack.append(cell);
    }
    ui.metrics.tHead?.prepend(rack);
  }
  const reportedRows = reported.flatMap(([, m]) => {
    const tr = el("tr", "reported");
    const name = el("td", "");
    const a = el("a", "", m.label);
    if (m.model_url) a.href = m.model_url;
    name.append(a, el("span", "sub", t(lang, "numbers.reported")));
    const dagger = (v: string): string => `${v} †`;
    tr.append(name, cell(m.params ?? "—"), cell(t(lang, "numbers.reportedMos")),
      cell(m.wer_all === undefined ? "—" : dagger(percent(lang, m.wer_all, 1))), cell("—"), cell("—"),
      cell(m.rtf === undefined ? "—" : dagger(`≈ ${number(lang, m.rtf, 0)}×`)), cell(m.license ?? "—", false));
    return [tr];
  });
  if (reportedRows.length) {
    const divider = el("tr", "divider");
    const td = el("td", "", t(lang, "numbers.reportedHead"));
    td.colSpan = 8;
    divider.append(td);
    reportedRows.unshift(divider);
  }
  ui.metrics.tBodies[0]!.replaceChildren(...rows.map(([key, m]) => {
    const tr = el("tr", key === "ours" ? "ours" : "");
    const name = el("td", "");
    if (m.model_url) {
      const a = el("a", "", m.label);
      a.href = m.model_url;
      name.append(a);
    } else {
      name.append(m.label);
    }
    if (key === "ours") name.append(el("span", "mark", t(lang, "numbers.ours")));
    const utmos = m.utmos === undefined ? "—" : number(lang, m.utmos, 2);
    tr.append(name, cell(m.params ?? "—"), cell(utmos, true, m.utmos === top.utmos),
      cell(m.wer_all === undefined ? "—" : percent(lang, m.wer_all), true, m.wer_all === top.wer),
      cell(m.wer_clean === undefined ? "—" : percent(lang, m.wer_clean), true, m.wer_clean === top.werc),
      cell(m.cer_clean === undefined ? "—" : percent(lang, m.cer_clean), true, m.cer_clean === top.cerc),
      cell(m.rtf === undefined ? "—" : `${number(lang, m.rtf, 0)}×`, true, m.rtf === top.rtf),
      cell(m.license ?? "—", false));
    return tr;
  }), ...reportedRows);
  if (metrics?.links?.length) {
    ui.links.replaceChildren(...metrics.links.map((l) => {
      const a = el("a", "", l.label);
      a.href = l.url;
      return a;
    }));
  }
}

async function json<T>(url: string): Promise<T | undefined> {
  try {
    const response = await fetch(url);
    return response.ok ? ((await response.json()) as T) : undefined;
  } catch {
    return undefined;
  }
}

setLang(lang);
showOptions();
autosize();
preview();
send({ type: "load", ...(Number(new URLSearchParams(location.search).get("threads")) > 0
  ? { threads: Number(new URLSearchParams(location.search).get("threads")) } : {}) });
void Promise.all([json<Sample[]>("/demo/samples.json"), json<Metrics>("/demo/metrics.json")]).then(([s, m]) => {
  samples = s ?? [];
  metrics = m;
  renderSamples();
  renderMetrics();
});
