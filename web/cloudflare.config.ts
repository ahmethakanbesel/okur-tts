import { defineConfig } from "cf/config";

// The demo page on Cloudflare (Pages on Workers): static assets built by Vite, served with public/_headers
// (cross-origin isolation for multithreaded WebAssembly, CSP, long caching). Deploy with `npm run deploy`.
export default defineConfig({
  worker: {
    name: "okur-tts",
    compatibilityDate: "2026-10-01",
    observability: { enabled: true },
    // Missing files must fail as 404s: a page served in place of a model file would be parsed as a model.
    assets: { notFoundHandling: "404-page" },
    domains: ["okur.dokuz.gen.tr"],
  },
});
