import { defaultClientConditions, defineConfig } from "vite";

import { cloudflare } from "@cloudflare/vite-plugin";

// Cross-origin isolation lets ONNX Runtime use threads (SharedArrayBuffer); production gets the same headers from
// public/_headers on Cloudflare (cloudflare.config.ts).
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

export default defineConfig({
  plugins: [cloudflare()],
  // onnxruntime-web without its own .wasm bundled in: it loads /ort/<version>/ (see scripts/assets.mjs).
  resolve: { conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions] },
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: "es" },
  build: { target: "es2023", assetsInlineLimit: 0, chunkSizeWarningLimit: 1024 },
});
