import { copyFile, mkdir, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Copy ONNX Runtime's WebAssembly artifacts into `public/ort` so the browser
 * loads them from our own origin.
 *
 * The default `wasmPaths` is a CDN. Being cross-origin, ONNX Runtime fetches its
 * loader as text and re-imports it through a `blob:` URL — a step a strict
 * `script-src` blocks, and whose failure is cached in ORT's backend registry so
 * that every later attempt, CPU included, reports "no available backend found".
 *
 * Copying at build time (rather than committing the ~35 MB) keeps these binaries
 * exactly in step with whatever ONNX Runtime version transformers.js resolves.
 */

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const target = join(web, "public", "ort");

// transformers.js pins its own ONNX Runtime, so prefer its nested copy.
const candidates = [
  join(web, "node_modules", "@huggingface", "transformers", "node_modules", "onnxruntime-web", "dist"),
  join(web, "node_modules", "onnxruntime-web", "dist"),
];

// Only the CPU/WASM builds are needed: speech and semantic inference run on
// WASM, and the WebGPU (jsep) build is deliberately not shipped.
const wanted = [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
];

const source = candidates.find((candidate) => existsSync(candidate));
if (!source) {
  console.error("[copyOnnxRuntime] No onnxruntime-web/dist found. Run npm install first.");
  process.exit(1);
}

await mkdir(target, { recursive: true });
const available = new Set(await readdir(source));
const missing = wanted.filter((name) => !available.has(name));
if (missing.length) {
  console.error(`[copyOnnxRuntime] Missing from ${source}: ${missing.join(", ")}`);
  process.exit(1);
}

await Promise.all(wanted.map((name) => copyFile(join(source, name), join(target, name))));
console.log(`[copyOnnxRuntime] Copied ${wanted.length} runtime files from ${source.replace(web, ".")}`);
