// Gather everything the site serves into public/: model graphs + config (from `okur export-web`), the frontend
// WebAssembly (from frontend-rs), ONNX Runtime's WebAssembly build. Content-hashed or versioned names, cached forever.
//   node scripts/assets.mjs <export dir> <frontend .wasm> [<demo dir>]
// With a demo dir (scripts/compare_all.sh output: metrics.json, samples.json, audio/), the comparison is published too.
import { createHash } from "node:crypto";
import { copyFileSync, cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

const [exportDir, frontendWasm, demoDir] = process.argv.slice(2);
if (!exportDir || !frontendWasm) throw new Error("usage: node scripts/assets.mjs <export dir> <frontend .wasm>");

const models = "public/models";
rmSync(models, { recursive: true, force: true });
mkdirSync(models, { recursive: true });
const config = JSON.parse(readFileSync(join(exportDir, "config.json"), "utf8"));
for (const file of Object.values(config.files)) copyFileSync(join(exportDir, file), join(models, file));

const wasm = readFileSync(frontendWasm);
const name = `frontend.${createHash("sha256").update(wasm).digest("hex").slice(0, 12)}.wasm`;
writeFileSync(join(models, name), wasm);
config.files.frontend = name;
config.bytes.frontend = wasm.length;
// The list of model files lives outside /models/ (cached forever) and is revalidated on every visit.
writeFileSync("public/models.json", JSON.stringify(config, null, 1));

// ONNX Runtime's threaded WebAssembly build, served as-is: its helper threads must load this standalone glue file,
// not our bundled worker.
const ortVersion = JSON.parse(readFileSync("node_modules/onnxruntime-web/package.json", "utf8")).version;
rmSync("public/ort", { recursive: true, force: true });
mkdirSync(`public/ort/${ortVersion}`, { recursive: true });
for (const f of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copyFileSync(join("node_modules/onnxruntime-web/dist", f), join(`public/ort/${ortVersion}`, f));
}
writeFileSync("src/generated.ts", `// written by scripts/assets.mjs\nexport const ORT_PATH = "/ort/${ortVersion}/";\n`);
console.log("models:", Object.values(config.files).join(", "), basename(frontendWasm), `| onnxruntime-web ${ortVersion}`);

// The comparison: rates as percentages, display names, and only the systems shown on the page. Speed for every system
// is its PyTorch or native runtime on the same 8 cores (ours: PyTorch, like EMA Lightning; the browser runs ONNX).
const SHOWN = {
  ours: { label: "Okur v1.0", params: "9,8 M", license: "Apache-2.0 / CC BY 4.0" },
  ema_lightning: { label: "EMA Lightning", params: "8,6 M", license: "Apache-2.0" },
  piper_dfki: { label: "Piper (dfki)", params: "≈ 16 M", license: "CC BY-NC-SA 4.0" },
  mms_tur: { label: "Meta MMS-TTS", params: "36 M", license: "CC BY-NC 4.0" },
};
if (demoDir) {
  rmSync("public/demo", { recursive: true, force: true });
  mkdirSync("public/demo", { recursive: true });
  cpSync(join(demoDir, "audio"), "public/demo/audio", { recursive: true });
  copyFileSync(join(demoDir, "samples.json"), "public/demo/samples.json");
  const raw = JSON.parse(readFileSync(join(demoDir, "metrics.json"), "utf8"));
  const pct = (x) => (typeof x === "number" ? 100 * x : undefined);
  const systems = Object.fromEntries(Object.entries(SHOWN).map(([key, shown]) => {
    const m = raw.systems[key];
    return [key, { ...shown, model_url: m.model_url ?? undefined, utmos: m.utmos, utmos_ci95: m.utmos_ci95,
      wer_all: pct(m.wer_all), wer_clean: pct(m.wer_clean), cer_clean: pct(m.cer_clean), rtf: m.rtf_torch ?? m.rtf }];
  }));
  // Published figures of another system, measured under its own protocol (8 kHz band-matched Whisper WER, human MOS,
  // RTX 4090 GPU speed). Shown in a separate row, never ranked against ours. FreyaTTS technical report, July 2026.
  systems.freya_small = { label: "FreyaTTS-small", params: "183 M", license: "Apache-2.0", reported: true,
    model_url: "https://huggingface.co/freyavoice/Freya-TTS", wer_all: 8.0, rtf: 9 };
  writeFileSync("public/demo/metrics.json", JSON.stringify({ systems }, null, 1));
  console.log("demo:", Object.keys(systems).join(", "));
}
