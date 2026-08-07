"use client";

import { useEffect, useState } from "react";
import { browserModels, formatBytes } from "@/config/browser-models";
import { detectCapabilities, type DeviceCapabilities } from "@/lib/inference/capabilities";
import { cacheSizeBytes, cacheStaticVisionModels, clearDownloadedModels, modelStatus, removeDownloadedModel } from "@/lib/inference/modelCache";

export function LocalModelManager() {
  const [capabilities, setCapabilities] = useState<DeviceCapabilities>();
  const [downloaded, setDownloaded] = useState<string[]>([]);
  const [storage, setStorage] = useState<number | null>(null);
  const [progress, setProgress] = useState<string>();
  useEffect(() => { void Promise.all([detectCapabilities(), cacheSizeBytes()]).then(([found, bytes]) => { setCapabilities(found); setStorage(bytes); setDownloaded(modelStatus().downloaded); }).catch(() => undefined); }, []);
  async function refresh() { setDownloaded(modelStatus().downloaded); setStorage(await cacheSizeBytes()); }
  async function warmVision() { try { setProgress("Downloading visual models to the browser cache…"); await cacheStaticVisionModels((loaded, total) => setProgress(`Caching visual models: ${formatBytes(loaded)} / ${formatBytes(total)}`)); await refresh(); setProgress("Visual models are ready for offline reuse."); } catch (error) { setProgress(error instanceof Error ? error.message : "The visual model download failed."); } }
  async function remove(id: keyof typeof browserModels) { try { setProgress(`Removing ${browserModels[id].label} from browser caches…`); await removeDownloadedModel(id); await refresh(); setProgress("Model cache updated."); } catch (error) { setProgress(error instanceof Error ? error.message : "The model could not be removed."); } }
  async function clear() { try { setProgress("Removing public model files from browser caches…"); await clearDownloadedModels(); await refresh(); setProgress("All downloaded model files were removed. Local interview recordings were not changed."); } catch (error) { setProgress(error instanceof Error ? error.message : "The model cache could not be cleared."); } }
  return <section className="setup-card"><div className="section-heading"><div><div className="eyebrow">Local model setup</div><h2>Browser capability and cache</h2></div>{capabilities && <span className="state-pill">{capabilities.recommendedTier}</span>}</div><p className="fine-print">Models download as public static files only. Your recording, transcript, and analysis never leave this browser.</p><div className="model-grid">{Object.values(browserModels).map((model) => <div key={model.id}><strong>{model.label}</strong><span>{formatBytes(model.estimatedBytes)}{model.optional ? " · optional" : ""}</span><small>{downloaded.includes(model.id) ? "Cached locally" : "Downloads on first use"}</small>{downloaded.includes(model.id) && <button className="text-button" onClick={() => void remove(model.id)}>Remove model</button>}</div>)}</div>{capabilities && <p className="notice">{capabilities.cpuThreads} logical CPU threads · WebAssembly {capabilities.wasm ? "available" : "unavailable"} · WebGPU {capabilities.webgpu ? "available" : "not detected"} · cache usage {storage === null ? "unavailable" : formatBytes(storage)}</p>}<div className="button-row"><button className="button secondary" onClick={() => void warmVision()}>Preload visual models</button><button className="button ghost" onClick={() => void clear()}>Clear all models</button></div>{progress && <p className="notice" role="status">{progress}</p>}</section>;
}
