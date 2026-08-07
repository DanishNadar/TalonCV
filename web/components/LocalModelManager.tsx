"use client";

import { useEffect, useState } from "react";
import { browserModels, formatBytes, type BrowserModelId } from "@/config/browser-models";
import { detectCapabilities, resolveInferenceMode, type DeviceCapabilities } from "@/lib/inference/capabilities";
import {
  cacheSizeBytes,
  cacheStaticVisionModels,
  clearDownloadedModels,
  modelStatus,
  removeDownloadedModel,
} from "@/lib/inference/modelCache";
import { LocalAIStatus, Meter, TechnicalBadge } from "@/components/ui/primitives";

const purposes: Record<BrowserModelId, string> = {
  speechFast: "Speech transcription · lite profile",
  speechBalanced: "Speech transcription · balanced profile",
  semantic: "Semantic relevance embeddings",
  vision: "Face and pose landmark models",
  coach: "Optional narrative coaching",
};

export function LocalModelManager() {
  const [capabilities, setCapabilities] = useState<DeviceCapabilities>();
  const [downloaded, setDownloaded] = useState<string[]>([]);
  const [storage, setStorage] = useState<number | null>(null);
  const [busy, setBusy] = useState<string>();
  const [progress, setProgress] = useState<string>();

  useEffect(() => {
    void Promise.all([detectCapabilities(), cacheSizeBytes()])
      .then(([found, bytes]) => {
        setCapabilities(found);
        setStorage(bytes);
        setDownloaded(modelStatus().downloaded);
      })
      .catch(() => undefined);
  }, []);

  async function refresh() {
    setDownloaded(modelStatus().downloaded);
    setStorage(await cacheSizeBytes());
  }

  async function warmVision() {
    setBusy("vision");
    try {
      setProgress("Downloading visual models to the browser cache…");
      await cacheStaticVisionModels((loaded, total) =>
        setProgress(`Caching visual models · ${formatBytes(loaded)} / ${formatBytes(total)}`),
      );
      await refresh();
      setProgress("Visual models are ready for offline reuse.");
    } catch (error) {
      setProgress(error instanceof Error ? error.message : "The visual model download failed.");
    } finally {
      setBusy(undefined);
    }
  }

  async function remove(id: BrowserModelId) {
    setBusy(id);
    try {
      setProgress(`Removing ${browserModels[id].label} from browser caches…`);
      await removeDownloadedModel(id);
      await refresh();
      setProgress("Model cache updated.");
    } catch (error) {
      setProgress(error instanceof Error ? error.message : "The model could not be removed.");
    } finally {
      setBusy(undefined);
    }
  }

  async function clear() {
    setBusy("all");
    try {
      setProgress("Removing public model files from browser caches…");
      await clearDownloadedModels();
      await refresh();
      setProgress("All downloaded model files were removed. Local interview recordings were not changed.");
    } catch (error) {
      setProgress(error instanceof Error ? error.message : "The model cache could not be cleared.");
    } finally {
      setBusy(undefined);
    }
  }

  const quota = capabilities?.storageQuotaBytes ?? null;
  const used = storage ?? capabilities?.storageUsageBytes ?? null;

  return (
    <div className="stack-6">
      <section className="panel accent-top">
        <div className="panel-header">
          <h2>Local AI models</h2>
          <div className="row">
            {capabilities ? <TechnicalBadge tone="info">{capabilities.recommendedTier} tier</TechnicalBadge> : null}
            <LocalAIStatus variant="compact" label="Cache" detail="Stored in this browser" />
          </div>
        </div>
        <div className="panel-body stack-4">
          <p className="fine-print">
            Models are downloaded as public static files and cached on this device. Your recording, transcript, and
            analysis are never sent to a model host.
          </p>
          <div className="model-list">
            {Object.values(browserModels).map((model) => {
              const cached = downloaded.includes(model.id);
              return (
                <article className="model-card" key={model.id}>
                  <div className="model-info">
                    <div className="name-row">
                      <h3>{model.label}</h3>
                      <TechnicalBadge tone={cached ? "success" : "neutral"} dot>
                        {cached ? "Cached" : "Not downloaded"}
                      </TechnicalBadge>
                      {model.optional ? <TechnicalBadge>Optional</TechnicalBadge> : null}
                    </div>
                    <p>{purposes[model.id]}</p>
                    <p className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
                      {formatBytes(model.estimatedBytes)} · {model.runtime}
                      {model.revision ? ` · ${model.revision.slice(0, 7)}` : ""}
                    </p>
                  </div>
                  <div className="model-actions">
                    {model.id === "vision" && !cached ? (
                      <button className="button secondary small" disabled={busy === "vision"} onClick={() => void warmVision()}>
                        {busy === "vision" ? "Downloading…" : "Download"}
                      </button>
                    ) : null}
                    {cached ? (
                      <button className="button ghost small" disabled={busy === model.id} onClick={() => void remove(model.id)}>
                        Remove
                      </button>
                    ) : null}
                  </div>
                </article>
              );
            })}
          </div>

          {progress ? (
            <p className="notice" role="status">
              <span aria-hidden="true">◆</span>
              <span>{progress}</span>
            </p>
          ) : null}

          <div className="button-row">
            <button className="button secondary" disabled={busy === "vision"} onClick={() => void warmVision()}>
              Preload visual models
            </button>
            <button className="button ghost" disabled={busy === "all"} onClick={() => void clear()}>
              Clear model cache
            </button>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Device and storage</h2>
          <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
            diagnostics
          </span>
        </div>
        <div className="panel-body stack-5">
          <div className="storage-bar">
            <Meter
              label="Browser storage used"
              value={quota && used ? Math.min(100, (used / quota) * 100) : null}
              display={used === null ? "Unavailable" : quota ? `${formatBytes(used)} / ${formatBytes(quota)}` : formatBytes(used)}
              caption="Includes cached model files and locally stored recordings"
              tone="info"
            />
          </div>
          <div className="readiness">
            <div className="readiness-row">
              <span />
              <span className="label">Logical CPU threads</span>
              <span className="value">{capabilities?.cpuThreads ?? "—"}</span>
            </div>
            <div className="readiness-row">
              <span />
              <span className="label">WebAssembly</span>
              <span className="value">{capabilities ? (capabilities.wasm ? "Available" : "Unavailable") : "—"}</span>
            </div>
            <div className="readiness-row">
              <span />
              <span className="label">WebGPU</span>
              <span className="value">{capabilities ? (capabilities.webgpu ? "Available" : "Not detected") : "—"}</span>
            </div>
            <div className="readiness-row">
              <span />
              <span className="label">Selected analysis mode</span>
              <span className="value">
                {capabilities ? (resolveInferenceMode("automatic", capabilities) === "webgpu" ? "WebGPU" : "CPU / WASM") : "—"}
              </span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
