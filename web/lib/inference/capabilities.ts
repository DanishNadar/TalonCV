import type { InferenceMode } from "@/types/local";

export interface DeviceCapabilities {
  ready: boolean;
  wasm: boolean;
  wasmSimd: boolean | null;
  wasmThreads: boolean;
  sharedArrayBuffer: boolean;
  crossOriginIsolated: boolean;
  webgpu: boolean;
  cpuThreads: number;
  memoryGb: number | null;
  storageQuotaBytes: number | null;
  storageUsageBytes: number | null;
  persistentStorage: boolean | null;
  mediaRecorder: boolean;
  audioContext: boolean;
  offscreenCanvas: boolean;
  recommendedTier: "lite" | "standard" | "accelerated";
}

function supportsWasmSimd(): boolean | null {
  try {
    return WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 9, 1, 7, 0, 65, 0, 253, 15, 11]));
  } catch { return null; }
}

export async function detectCapabilities(): Promise<DeviceCapabilities> {
  if (typeof window === "undefined") throw new Error("Capability detection requires a browser.");
  const nav = navigator as Navigator & { deviceMemory?: number; gpu?: unknown };
  const estimate = await navigator.storage?.estimate?.().catch(() => undefined);
  const persistentStorage = navigator.storage?.persisted ? await navigator.storage.persisted().catch(() => null) : null;
  const cpuThreads = Math.max(1, nav.hardwareConcurrency || 1);
  const memoryGb = typeof nav.deviceMemory === "number" ? nav.deviceMemory : null;
  const webgpu = Boolean(nav.gpu);
  const tier = webgpu && (memoryGb === null || memoryGb >= 8) ? "accelerated" : cpuThreads >= 4 && (memoryGb === null || memoryGb >= 4) ? "standard" : "lite";
  return {
    ready: typeof WebAssembly !== "undefined" && typeof Worker !== "undefined" && Boolean(navigator.mediaDevices?.getUserMedia),
    wasm: typeof WebAssembly !== "undefined", wasmSimd: supportsWasmSimd(), wasmThreads: typeof SharedArrayBuffer !== "undefined" && crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined", crossOriginIsolated, webgpu, cpuThreads, memoryGb,
    storageQuotaBytes: estimate?.quota ?? null, storageUsageBytes: estimate?.usage ?? null, persistentStorage,
    mediaRecorder: typeof MediaRecorder !== "undefined", audioContext: typeof AudioContext !== "undefined",
    offscreenCanvas: typeof OffscreenCanvas !== "undefined", recommendedTier: tier,
  };
}

export function resolveInferenceMode(requested: InferenceMode, capabilities: DeviceCapabilities): "wasm" | "webgpu" {
  return requested === "webgpu" && capabilities.webgpu ? "webgpu" : requested === "automatic" && capabilities.webgpu && capabilities.recommendedTier === "accelerated" ? "webgpu" : "wasm";
}

export function recommendedWorkerThreads(cpuThreads: number): number { return cpuThreads <= 2 ? 1 : cpuThreads <= 4 ? 2 : Math.min(4, Math.max(3, Math.floor(cpuThreads / 2))); }
