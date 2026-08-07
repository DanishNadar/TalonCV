"use client";

import { browserModels, formatBytes, type BrowserModelId } from "@/config/browser-models";
import { LocalAIStatus, TechnicalBadge, TechnicalPanel } from "@/components/ui/primitives";
import type { AnalysisProgress } from "@/lib/inference/localAnalysis";

export interface StageDefinition { id: string; name: string; runtime: string }

/** Display stages, in the order the pipeline actually executes them. */
export const stages: StageDefinition[] = [
  { id: "preparingMedia", name: "Preparing media", runtime: "Web Audio decode" },
  { id: "speech", name: "Speech transcription", runtime: "Whisper Tiny · ONNX Runtime Web" },
  { id: "vocal", name: "Vocal analysis", runtime: "Deterministic signal DSP" },
  { id: "semantic", name: "Semantic analysis", runtime: "MiniLM embeddings" },
  { id: "visual", name: "Visual analysis", runtime: "YOLO ONNX · MediaPipe" },
  { id: "alignment", name: "Multimodal alignment", runtime: "Timestamp overlap rules" },
  { id: "scoring", name: "Explainable scoring", runtime: "Weighted rubric" },
  { id: "report", name: "Report generation", runtime: "Deterministic template" },
];

/** Maps a raw worker progress event onto one of the display stages. */
export function resolveStage(progress: AnalysisProgress | undefined): string {
  if (!progress) return "preparingMedia";
  const { stage, modelId } = progress;
  if (stage === "downloadingModel") {
    if (modelId === "semantic") return "semantic";
    if (modelId === "vision") return "visual";
    return "speech";
  }
  if (stage === "loadingSpeech" || stage === "transcribing") return "speech";
  if (stage === "analyzingAudio") return "vocal";
  if (stage === "loadingSemantic" || stage === "analyzingSemantic" || stage === "analyzingResponse") return "semantic";
  if (stage === "loadingVision" || stage === "analyzingVisual") return "visual";
  if (stage === "aligningEvidence") return "alignment";
  if (stage === "calculatingScores") return "scoring";
  if (stage === "complete") return "report";
  return "preparingMedia";
}

export interface ModelDownload { id: BrowserModelId; loaded: number; total: number }

/** Raw worker messages name individual weight files. Novice users get the model
 *  name instead; the raw string stays in the technical details panel. */
function humanMessage(progress: AnalysisProgress | undefined): string {
  if (!progress) return "Starting the local pipeline";
  if (progress.stage === "downloadingModel") {
    const model = progress.modelId ? browserModels[progress.modelId as BrowserModelId] : undefined;
    return model ? `Downloading ${model.label} to this device` : "Downloading local model files";
  }
  return progress.message;
}

export function AnalysisRunView({
  progress,
  downloads,
  question,
  onCancel,
}: {
  progress?: AnalysisProgress;
  downloads: ModelDownload[];
  question: string;
  onCancel: () => void;
}) {
  const activeStage = resolveStage(progress);
  const activeIndex = Math.max(0, stages.findIndex((stage) => stage.id === activeStage));
  // Worker stages report their own scales, so overall completion is derived from
  // pipeline position rather than the raw per-stage percentage.
  const percent = Math.round(((activeIndex + 1) / stages.length) * 100);
  const tracked = downloads.filter((download) => download.total > 0);

  return (
    <div className="progress-page">
      <div className="progress-headline">
        <span className="eyebrow">Local analysis · stage {String(activeIndex + 1).padStart(2, "0")} of {stages.length}</span>
        <h1>{stages[activeIndex].name}</h1>
        <p className="lede">{humanMessage(progress)}</p>
        <span className="fine-print">{question}</span>
      </div>

      <div className="large-progress" role="progressbar" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100} aria-label="Overall analysis progress">
        <span style={{ width: `${percent}%` }} />
      </div>

      <div className="progress-split">
        <section className="panel accent-top">
          <div className="panel-header">
            <h2>Pipeline</h2>
            <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
              {Math.round(percent)}%
            </span>
          </div>
          <div>
            {stages.map((stage, index) => {
              const state = index < activeIndex ? "done" : index === activeIndex ? "active" : "pending";
              return (
                <div className="analysis-stage" data-state={state} key={stage.id}>
                  <span className="stage-idx">{String(index + 1).padStart(2, "0")}</span>
                  <span className="stage-name">{stage.name}</span>
                  <span className="stage-mark" aria-hidden="true">
                    {state === "done" ? "✓" : <span className="ring" />}
                  </span>
                  <span className="visually-hidden">
                    {state === "done" ? "complete" : state === "active" ? "in progress" : "waiting"}
                  </span>
                </div>
              );
            })}
          </div>
        </section>

        <div className="stack-4">
          {tracked.length ? (
            <section className="panel">
              <div className="panel-header">
                <h2>Preparing local AI</h2>
                <TechnicalBadge tone="info" dot>
                  First use
                </TechnicalBadge>
              </div>
              <div>
                {tracked.map((download) => {
                  const model = browserModels[download.id];
                  const ratio = Math.min(100, (download.loaded / download.total) * 100);
                  const ready = download.loaded >= download.total;
                  return (
                    <div className="model-load-row" key={download.id}>
                      <div className="meter-head">
                        <span>{model.label}</span>
                        <strong>
                          {ready ? "Ready" : `${formatBytes(download.loaded)} / ${formatBytes(download.total)}`}
                        </strong>
                      </div>
                      <div className="meter-track">
                        <span className={`meter-fill ${ready ? "success" : ""}`} style={{ width: `${ratio}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="panel-body">
                <p className="fine-print">
                  These model files are cached on this device for future sessions. No interview content is being uploaded.
                </p>
              </div>
            </section>
          ) : null}

          <section className="panel">
            <div className="panel-header">
              <h2>Runtime</h2>
              <LocalAIStatus variant="compact" label="On-device" detail="No egress" />
            </div>
            <div className="panel-body stack-4">
              <div className="readiness">
                <div className="readiness-row">
                  <span />
                  <span className="label">Active stage</span>
                  <span className="value">{stages[Math.max(0, activeIndex)]?.name ?? "—"}</span>
                </div>
                <div className="readiness-row">
                  <span />
                  <span className="label">Model / runtime</span>
                  <span className="value">{stages[Math.max(0, activeIndex)]?.runtime ?? "—"}</span>
                </div>
                <div className="readiness-row">
                  <span />
                  <span className="label">Interview upload</span>
                  <span className="value">None</span>
                </div>
              </div>
              <button className="button ghost" onClick={onCancel}>
                Cancel analysis
              </button>
            </div>
          </section>

          <TechnicalPanel label="Technical details">
            <pre>
              {JSON.stringify(
                {
                  rawStage: progress?.stage ?? null,
                  rawMessage: progress?.message ?? null,
                  displayStage: activeStage,
                  rawPercent: progress?.progress ?? null,
                  overallPercent: percent,
                  modelId: progress?.modelId ?? null,
                  loadedBytes: progress?.loadedBytes ?? null,
                  totalBytes: progress?.totalBytes ?? null,
                },
                null,
                2,
              )}
            </pre>
          </TechnicalPanel>
        </div>
      </div>
    </div>
  );
}
