import { env } from "@huggingface/transformers";

/**
 * Point ONNX Runtime Web at same-origin WebAssembly artifacts.
 *
 * The default `wasmPaths` is a CDN. Because that is cross-origin, ORT fetches
 * its loader as text and re-imports it through a `blob:` URL, which a strict
 * `script-src` blocks. Worse, when that import fails ORT caches the failure in
 * its backend registry, so every later attempt — including CPU ones — reports
 * the same "no available backend found" error and no fallback can recover.
 *
 * Serving the artifacts from our own origin removes the blob indirection
 * entirely, so transcription no longer depends on the content-security policy.
 */
export function configureLocalOnnxRuntime(): void {
  const wasm = env.backends?.onnx?.wasm as
    | { wasmPaths?: string; proxy?: boolean; numThreads?: number }
    | undefined;
  if (!wasm) return;
  wasm.wasmPaths = "/ort/";
  // Threads need SharedArrayBuffer, which needs cross-origin isolation we do not
  // set. Asking for one thread avoids ORT spawning a blob-backed thread worker.
  wasm.numThreads = 1;
  wasm.proxy = false;
}
