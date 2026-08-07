"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { createLocalAnalysis, type AnalysisProgress } from "@/lib/inference/localAnalysis";
import { detectCapabilities, resolveInferenceMode } from "@/lib/inference/capabilities";
import { generateLocalCoaching } from "@/lib/inference/coaching/localTextGenerator";
import { markModelReady } from "@/lib/inference/modelCache";
import type { BrowserModelId } from "@/config/browser-models";
import { getAnalysis, saveAnalysis } from "@/lib/storage/artifactStore";
import { getRecording } from "@/lib/storage/mediaStore";
import { buildSessionBundle, bundleFilename, downloadBlob, downloadText } from "@/lib/storage/sessionExport";
import { getSession, saveSession } from "@/lib/storage/sessionStore";
import { AnalysisRunView, type ModelDownload } from "@/components/analysis/AnalysisRunView";
import { ResultTabPanel, reviewTabs, type Tab } from "@/components/analysis/ResultTabs";
import { LocalAIStatus, TechnicalBadge } from "@/components/ui/primitives";
import type { LocalAnalysis, LocalSession } from "@/types/local";

export { reviewTabs } from "@/components/analysis/ResultTabs";

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : {};

export function seekVideoElement(video: HTMLVideoElement | null, timestamp: number): boolean {
  if (!video || !Number.isFinite(timestamp)) return false;
  video.currentTime = Math.max(0, timestamp);
  void video.play().catch(() => undefined);
  return true;
}

export function LocalAnalysisWorkspace({ sessionId, autoRun = false }: { sessionId: string; autoRun?: boolean }) {
  const playerRef = useRef<HTMLVideoElement>(null);
  const recordingUrlRef = useRef<string | undefined>(undefined);
  const controllerRef = useRef<ReturnType<typeof createLocalAnalysis> | undefined>(undefined);
  const autoRunRef = useRef(autoRun);

  const [session, setSession] = useState<LocalSession>();
  const [analysis, setAnalysis] = useState<LocalAnalysis>();
  const [recordingUrl, setRecordingUrl] = useState<string>();
  const [tab, setTab] = useState<Tab>("Overview");
  const [progress, setProgress] = useState<AnalysisProgress>();
  const [downloads, setDownloads] = useState<ModelDownload[]>([]);
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [packaging, setPackaging] = useState(false);
  const [coachingBusy, setCoachingBusy] = useState(false);
  const [modelMode, setModelMode] = useState<"automatic" | "cpu" | "webgpu">("automatic");

  const load = useCallback(async () => {
    try {
      const [current, storedAnalysis, recording] = await Promise.all([
        getSession(sessionId),
        getAnalysis(sessionId),
        getRecording(sessionId),
      ]);
      setSession(current);
      setAnalysis(storedAnalysis);
      if (recording) {
        if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
        const url = URL.createObjectURL(recording);
        recordingUrlRef.current = url;
        setRecordingUrl(url);
      }
      if (!current) setError("This local session is not available in this browser.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load local session.");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => {
      window.clearTimeout(timer);
      if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
    };
  }, [load]);

  const trackProgress = useCallback((next: AnalysisProgress) => {
    if (next.modelId && (next.progress >= 100 || next.stage === "transcribing" || next.stage === "analyzingSemantic")) {
      markModelReady(next.modelId as BrowserModelId);
    }
    if (next.modelId && typeof next.totalBytes === "number" && next.totalBytes > 0) {
      const id = next.modelId as BrowserModelId;
      const loaded = next.loadedBytes ?? 0;
      const total = next.totalBytes;
      setDownloads((current) => {
        const existing = current.find((item) => item.id === id);
        if (!existing) return [...current, { id, loaded, total }];
        if (loaded <= existing.loaded && total === existing.total) return current;
        return current.map((item) => (item.id === id ? { id, loaded, total } : item));
      });
    }
    setProgress(next);
  }, []);

  const runAnalysis = useCallback(async () => {
    const current = session ?? (await getSession(sessionId));
    if (!current) return;
    const recording = await getRecording(current.id);
    if (!recording) {
      setError("The locally stored recording is unavailable.");
      return;
    }
    setRunning(true);
    setError(undefined);
    setDownloads([]);
    setProgress({ stage: "preparingMedia", progress: 1, message: "Preparing local media" });
    const working = { ...current, analysisState: "preparing" as const, error: undefined };
    await saveSession(working);
    setSession(working);
    try {
      const capabilities = await detectCapabilities();
      const controller = createLocalAnalysis(
        working,
        recording,
        {
          provider: resolveInferenceMode(modelMode, capabilities),
          profile: capabilities.recommendedTier === "lite" ? "fast" : "balanced",
          visualFps: capabilities.recommendedTier === "lite" ? 1 : 2,
        },
        trackProgress,
      );
      controllerRef.current = controller;
      const completed = await controller.promise;
      await saveAnalysis(working.id, completed);
      const score = asRecord(asRecord(completed.scores).scores).overallInterviewPracticeDelivery;
      const updated = {
        ...working,
        analysisState: "complete" as const,
        analysisVersion: completed.analysisVersion,
        modelVersions: completed.modelVersions,
        overallScore: typeof asRecord(score).score === "number" ? (asRecord(score).score as number) : undefined,
      };
      await saveSession(updated);
      setSession(updated);
      setAnalysis(completed);
      setTab("Overview");
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Local analysis failed.";
      const failed = { ...working, analysisState: "failed" as const, error: message };
      await saveSession(failed);
      setSession(failed);
      setError(message);
    } finally {
      controllerRef.current = undefined;
      setRunning(false);
    }
  }, [modelMode, session, sessionId, trackProgress]);

  // Honours ?rerun=1 from the history page, exactly once per mount.
  useEffect(() => {
    if (!autoRunRef.current || loading || !session || running) return;
    autoRunRef.current = false;
    void runAnalysis();
  }, [loading, running, runAnalysis, session]);

  async function enhanceCoaching() {
    if (!analysis) return;
    setCoachingBusy(true);
    try {
      const result = await generateLocalCoaching(
        { report: analysis.report, scores: analysis.scores, moments: analysis.moments, warnings: analysis.warnings },
        modelMode,
      );
      const next = { ...analysis, localCoaching: result };
      await saveAnalysis(sessionId, next);
      setAnalysis(next);
    } finally {
      setCoachingBusy(false);
    }
  }

  async function downloadBundle() {
    if (!analysis || !session) return;
    setPackaging(true);
    try {
      downloadBlob(await buildSessionBundle(session, analysis), bundleFilename(session));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The session bundle could not be created.");
    } finally {
      setPackaging(false);
    }
  }

  async function downloadArtifact(kind: "report" | "transcript" | "analysis" | "recording") {
    if (!analysis || !session) return;
    if (kind === "report") return downloadText(analysis.report, `taloncv-report-${session.id}.md`, "text/markdown");
    if (kind === "transcript") return downloadText(analysis.transcript.text, `taloncv-transcript-${session.id}.txt`, "text/plain");
    if (kind === "analysis") {
      return downloadText(JSON.stringify(analysis, null, 2), `taloncv-analysis-${session.id}.json`, "application/json");
    }
    const recording = await getRecording(session.id);
    if (recording) {
      const extension = recording.type.split("/")[1]?.split(";")[0] || "webm";
      downloadBlob(recording, `taloncv-recording-${session.id}.${extension}`);
    }
  }

  const seek = (timestamp: number) => {
    seekVideoElement(playerRef.current, timestamp);
  };

  if (loading) {
    return (
      <section className="shell centered-state">
        <span className="spinner" aria-hidden="true" />
        <p>Opening local review…</p>
      </section>
    );
  }

  if (!session) {
    return (
      <section className="shell centered-state">
        <h1>Local review not found</h1>
        <p>{error || "This browser may have cleared its local TalonCV storage."}</p>
        <Link className="button primary" href="/dashboard">
          Back to history
        </Link>
      </section>
    );
  }

  if (running) {
    return (
      <div className="shell">
        <AnalysisRunView
          progress={progress}
          downloads={downloads}
          question={session.context.interviewQuestion}
          onCancel={() => controllerRef.current?.cancel()}
        />
      </div>
    );
  }

  return (
    <div className="shell results-page">
      <header className="results-heading">
        <div>
          <div className="eyebrow">Local interview review</div>
          <h1>{session.context.interviewQuestion}</h1>
          <p className="sub">
            {session.context.targetRole || "General interview practice"} ·{" "}
            {new Date(session.createdAt).toLocaleString()} · stored only in this browser
          </p>
        </div>
        <div className="results-controls">
          <LocalAIStatus variant="compact" />
          <label>
            <span className="visually-hidden">Inference mode</span>
            <select value={modelMode} disabled={running} onChange={(event) => setModelMode(event.target.value as typeof modelMode)}>
              <option value="automatic">Automatic</option>
              <option value="cpu">CPU / WASM</option>
              <option value="webgpu">WebGPU when available</option>
            </select>
          </label>
        </div>
      </header>

      {recordingUrl ? (
        <div className="result-player">
          <video ref={playerRef} controls preload="metadata" src={recordingUrl} />
          <div className="player-foot">
            <p>Replay stays local to this browser.</p>
            <TechnicalBadge plain>{session.recording?.mimeType ?? "media"}</TechnicalBadge>
          </div>
        </div>
      ) : null}

      {error ? (
        <p className="notice error" role="alert">
          <span aria-hidden="true">!</span>
          <span>{error}</span>
        </p>
      ) : null}

      {!analysis ? (
        <section className="centered-state">
          <span className="eyebrow">Stage 04</span>
          <h1>Ready for a local review</h1>
          <p>
            Whisper, MiniLM, MediaPipe, YOLO ONNX, deterministic cue rules, and explainable scoring all execute in this
            browser. Public model files download once and are cached; the recording is never uploaded.
          </p>
          <div className="button-row">
            <button className="button primary large" onClick={() => void runAnalysis()}>
              Run local multimodal analysis
            </button>
          </div>
        </section>
      ) : (
        <>
          <div className="tabs" role="tablist" aria-label="Review sections">
            {reviewTabs.map((item) => (
              <button
                key={item}
                role="tab"
                id={`tab-${item.replaceAll(" ", "-")}`}
                aria-selected={tab === item}
                aria-controls="review-panel"
                onClick={() => setTab(item)}
              >
                {item}
              </button>
            ))}
          </div>
          <section className="tab-panel" role="tabpanel" id="review-panel" aria-label={tab} tabIndex={-1}>
            <ResultTabPanel
              tab={tab}
              analysis={analysis}
              seek={seek}
              onDownloadBundle={() => void downloadBundle()}
              onDownloadArtifact={(kind) => void downloadArtifact(kind)}
              onGenerateCoaching={() => void enhanceCoaching()}
              coachingBusy={coachingBusy}
              packaging={packaging}
              onRerun={() => void runAnalysis()}
            />
          </section>
        </>
      )}
    </div>
  );
}
