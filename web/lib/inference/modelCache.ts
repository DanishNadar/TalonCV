import { browserModels, type BrowserModelId } from "@/config/browser-models";

const cacheName = "taloncv-static-models-v1";
const statusKey = "taloncv-model-status";
export interface ModelCacheStatus { downloaded: BrowserModelId[]; updatedAt: string; }

function readStatus(): ModelCacheStatus {
  try { return JSON.parse(localStorage.getItem(statusKey) || "{\"downloaded\":[],\"updatedAt\":\"\"}") as ModelCacheStatus; } catch { return { downloaded: [], updatedAt: "" }; }
}
function writeStatus(status: ModelCacheStatus) { localStorage.setItem(statusKey, JSON.stringify(status)); }
export function modelStatus(): ModelCacheStatus { return typeof window === "undefined" ? { downloaded: [], updatedAt: "" } : readStatus(); }
export function markModelReady(id: BrowserModelId): void {
  const status = readStatus(); if (!status.downloaded.includes(id)) status.downloaded.push(id); status.updatedAt = new Date().toISOString(); writeStatus(status);
}
function unmarkModel(id: BrowserModelId): void { const status = readStatus(); status.downloaded = status.downloaded.filter((item) => item !== id); status.updatedAt = new Date().toISOString(); writeStatus(status); }
export async function cacheStaticVisionModels(onProgress?: (loaded: number, total: number) => void, signal?: AbortSignal): Promise<void> {
  const model = browserModels.vision;
  const cache = await caches.open(cacheName); let loaded = 0; const total = model.estimatedBytes;
  for (const file of model.files || []) {
    if (await cache.match(file.url)) { loaded += file.bytes; onProgress?.(loaded, total); continue; }
    const response = await fetch(file.url, { signal, cache: "force-cache" });
    if (!response.ok) throw new Error(`Could not download visual model (${response.status}).`);
    await cache.put(file.url, response.clone()); loaded += file.bytes; onProgress?.(loaded, total);
  }
  markModelReady("vision");
}
export async function cacheSizeBytes(): Promise<number | null> {
  if (typeof navigator === "undefined" || !navigator.storage?.estimate) return null;
  return (await navigator.storage.estimate()).usage ?? null;
}
export async function clearDownloadedModels(): Promise<void> {
  await caches.delete(cacheName);
  for (const name of await caches.keys()) {
    const cache = await caches.open(name);
    for (const request of await cache.keys()) {
      const url = new URL(request.url);
      if (url.hostname === "huggingface.co" || url.hostname.endsWith("huggingface.co") || url.hostname === "storage.googleapis.com" && url.pathname.includes("mediapipe-models")) await cache.delete(request);
    }
  }
  localStorage.removeItem(statusKey);
}

export async function removeDownloadedModel(id: BrowserModelId): Promise<void> {
  const model = browserModels[id];
  if (id === "vision") {
    const cache = await caches.open(cacheName);
    await Promise.all((model.files || []).map((file) => cache.delete(file.url)));
  } else if (model.model) {
    for (const name of await caches.keys()) {
      const cache = await caches.open(name);
      for (const request of await cache.keys()) if (decodeURIComponent(request.url).includes(model.model)) await cache.delete(request);
    }
  }
  unmarkModel(id);
}
