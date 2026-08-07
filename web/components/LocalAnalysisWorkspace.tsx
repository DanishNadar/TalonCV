"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { zipSync, strToU8 } from "fflate";
import { createLocalAnalysis, type AnalysisProgress } from "@/lib/inference/localAnalysis";
import { detectCapabilities, resolveInferenceMode } from "@/lib/inference/capabilities";
import { deterministicCoach } from "@/lib/inference/coaching/deterministicCoach";
import { generateLocalCoaching } from "@/lib/inference/coaching/localTextGenerator";
import { markModelReady } from "@/lib/inference/modelCache";
import type { BrowserModelId } from "@/config/browser-models";
import { getAnalysis, saveAnalysis } from "@/lib/storage/artifactStore";
import { getRecording } from "@/lib/storage/mediaStore";
import { getSession, saveSession } from "@/lib/storage/sessionStore";
import type { EvidenceEvent, LocalAnalysis, LocalSession } from "@/types/local";

export const reviewTabs = ["Overview", "Transcript", "Answer Quality", "Vocal Delivery", "Visual Cues", "Multimodal Moments", "Full Report", "Downloads"] as const;
type Tab = (typeof reviewTabs)[number];
const asRecord = (value: unknown): Record<string, unknown> => value && typeof value === "object" ? value as Record<string, unknown> : {};
const number = (value: unknown): string => typeof value === "number" && Number.isFinite(value) ? value.toFixed(1) : "Unavailable";
const time = (value: unknown): string => typeof value === "number" && Number.isFinite(value) ? `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}` : "—";

export function seekVideoElement(video: HTMLVideoElement | null, timestamp: number): boolean {
  if (!video || !Number.isFinite(timestamp)) return false;
  video.currentTime = Math.max(0, timestamp);
  void video.play().catch(() => undefined);
  return true;
}

function EventList({ events, seek }: { events: EvidenceEvent[]; seek: (time: number) => void }) {
  if (!events.length) return <p className="notice">No reliable timestamped evidence was available for this modality.</p>;
  return <div className="evidence-list">{events.map((event, index) => <article key={`${event.eventType}-${event.startTime}-${index}`}><button className="timestamp-button" onClick={() => seek(event.startTime)}>{time(event.startTime)}</button><div><strong>{event.eventType.replaceAll(/([A-Z])/g, " $1")}</strong><p>{event.explanation}</p><small>{event.reliability || "medium"} reliability · {event.provenance || "rule"} evidence</small></div></article>)}</div>;
}

function ScoreGrid({ analysis }: { analysis: LocalAnalysis }) {
  const scores = asRecord(asRecord(analysis.scores).scores);
  return <div className="score-grid">{Object.entries(scores).map(([name, raw]) => { const score = asRecord(raw); return <div key={name}><span>{name.replaceAll(/([A-Z])/g, " $1")}</span><strong>{number(score.score)}</strong><small>{String(score.rating || "Unavailable")}</small></div>; })}</div>;
}

function Overview({ analysis, seek }: { analysis: LocalAnalysis; seek: (time: number) => void }) {
  const scores = asRecord(asRecord(analysis.scores).scores);
  const overall = asRecord(scores.overallInterviewPracticeDelivery);
  const coach = deterministicCoach(analysis);
  return <div className="tab-stack"><section className="score-hero"><div><span>Overall practice score</span><strong>{number(overall.score)}</strong><em>{String(overall.rating || "Unavailable")}</em></div><p>Explainable practice coaching based only on observable local recording evidence. It is not an employment, personality, or suitability assessment.</p></section><ScoreGrid analysis={analysis} /><section className="two-column"><article className="evidence-card"><h2>Prioritized practice</h2><ol className="recommendations">{coach.map((item) => <li key={item}>{item}</li>)}</ol></article><article className="evidence-card"><h2>Coverage</h2><p>{analysis.mediaInfo.hasAudio ? "Audio analyzed locally." : "No usable audio."} {analysis.mediaInfo.hasVideo ? "Video analyzed locally." : "No usable video."} Transcript words: {String(asRecord(analysis.responseAnalysis.metrics).wordCount ?? 0)}.</p></article></section><section><h2>Replayable multimodal moments</h2><Moments moments={analysis.moments} seek={seek} /></section></div>;
}

function Moments({ moments, seek }: { moments: Array<Record<string, unknown>>; seek: (time: number) => void }) {
  if (!moments.length) return <p className="notice">No cross-modal moments met the evidence rules. Individual visual and audio events remain available in their tabs.</p>;
  return <div className="moment-list">{moments.map((moment, index) => <article key={`${String(moment.alignmentCategory)}-${index}`}><div><button className="timestamp-button" onClick={() => seek(Number(moment.startTime))}>{time(moment.startTime)}</button><span className={`classification ${String(moment.classification || "context")}`}>{String(moment.classification || "context")}</span><em>{String(moment.alignmentCategory || "aligned evidence")}</em></div><blockquote>{String(moment.transcriptExcerpt || "No overlapping transcript excerpt.")}</blockquote><p>{String(moment.explanation || "")}</p><p><strong>Practice:</strong> {String(moment.coachingRecommendation || "Replay this timestamp in context.")}</p></article>)}</div>;
}

export function LocalAnalysisWorkspace({ sessionId }: { sessionId: string }) {
  const playerRef = useRef<HTMLVideoElement>(null);
  const recordingUrlRef = useRef<string | undefined>(undefined);
  const controllerRef = useRef<ReturnType<typeof createLocalAnalysis> | undefined>(undefined);
  const [session, setSession] = useState<LocalSession>();
  const [analysis, setAnalysis] = useState<LocalAnalysis>();
  const [recordingUrl, setRecordingUrl] = useState<string>();
  const [tab, setTab] = useState<Tab>("Overview");
  const [progress, setProgress] = useState<AnalysisProgress>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [modelMode, setModelMode] = useState<"automatic" | "cpu" | "webgpu">("automatic");

  const load = useCallback(async () => {
    try {
      const [current, storedAnalysis, recording] = await Promise.all([getSession(sessionId), getAnalysis(sessionId), getRecording(sessionId)]);
      setSession(current);
      setAnalysis(storedAnalysis);
      if (recording) { if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current); const url = URL.createObjectURL(recording); recordingUrlRef.current = url; setRecordingUrl(url); }
      if (!current) setError("This local session is not available in this browser.");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not load local session."); }
    finally { setLoading(false); }
  }, [sessionId]);

  useEffect(() => { const timer = window.setTimeout(() => void load(), 0); return () => { window.clearTimeout(timer); if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current); }; }, [load]);

  async function runAnalysis() {
    if (!session) return;
    const recording = await getRecording(session.id);
    if (!recording) { setError("The locally stored recording is unavailable."); return; }
    setRunning(true); setError(undefined); setProgress({ stage: "preparingMedia", progress: 1, message: "Preparing local media" });
    const working = { ...session, analysisState: "preparing" as const, error: undefined };
    await saveSession(working); setSession(working);
    try {
      const capabilities = await detectCapabilities();
      const controller = createLocalAnalysis(working, recording, { provider: resolveInferenceMode(modelMode, capabilities), profile: capabilities.recommendedTier === "lite" ? "fast" : "balanced", visualFps: capabilities.recommendedTier === "lite" ? 1 : 2 }, (next) => { if (next.modelId && (next.progress >= 100 || next.stage === "transcribing" || next.stage === "analyzingSemantic")) markModelReady(next.modelId as BrowserModelId); setProgress(next); });
      controllerRef.current = controller;
      const completed = await controller.promise;
      await saveAnalysis(working.id, completed);
      const score = asRecord(asRecord(completed.scores).scores).overallInterviewPracticeDelivery;
      const updated = { ...working, analysisState: "complete" as const, analysisVersion: completed.analysisVersion, modelVersions: completed.modelVersions, overallScore: typeof asRecord(score).score === "number" ? asRecord(score).score as number : undefined };
      await saveSession(updated);
      setSession(updated); setAnalysis(completed); setTab("Overview");
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Local analysis failed.";
      const failed = { ...working, analysisState: "failed" as const, error: message };
      await saveSession(failed); setSession(failed); setError(message);
    } finally { controllerRef.current = undefined; setRunning(false); }
  }

  async function enhanceCoaching() {
    if (!analysis) return;
    const result = await generateLocalCoaching({ report: analysis.report, scores: analysis.scores, moments: analysis.moments, warnings: analysis.warnings }, modelMode);
    const next = { ...analysis, localCoaching: result };
    await saveAnalysis(sessionId, next); setAnalysis(next);
  }

  async function downloadBundle() {
    if (!analysis || !session) return;
    const recording = await getRecording(session.id);
    const extension = recording?.type.split("/")[1]?.split(";")[0] || "webm";
    const files: Record<string, Uint8Array> = { "report.md": strToU8(analysis.report), "analysis.json": strToU8(JSON.stringify(analysis, null, 2)), "session.json": strToU8(JSON.stringify(session, null, 2)), "transcript.txt": strToU8(analysis.transcript.text), "transcript.json": strToU8(JSON.stringify(analysis.transcript, null, 2)), "audio_features.json": strToU8(JSON.stringify(analysis.audioFeatures, null, 2)), "audio_events.json": strToU8(JSON.stringify(analysis.audioEvents, null, 2)), "response_analysis.json": strToU8(JSON.stringify(analysis.responseAnalysis, null, 2)), "semantic_analysis.json": strToU8(JSON.stringify(analysis.semanticAnalysis, null, 2)), "visual_features.json": strToU8(JSON.stringify(analysis.visualFeatures, null, 2)), "visual_events.json": strToU8(JSON.stringify(analysis.visualEvents, null, 2)), "multimodal.json": strToU8(JSON.stringify(analysis.moments, null, 2)), "scores.json": strToU8(JSON.stringify(analysis.scores, null, 2)) };
    if (analysis.localCoaching?.available && analysis.localCoaching.text) files["coaching.md"] = strToU8(analysis.localCoaching.text);
    if (recording) files[`recording.${extension}`] = new Uint8Array(await recording.arrayBuffer());
    const blob = new Blob([zipSync(files)], { type: "application/zip" });
    const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `taloncv-local-${session.id}.zip`; link.click(); URL.revokeObjectURL(link.href);
  }

  const seek = (timestamp: number) => { seekVideoElement(playerRef.current, timestamp); };
  if (loading) return <section className="shell centered-state"><span className="spinner" /><p>Opening local review…</p></section>;
  if (!session) return <section className="shell centered-state"><h1>Local review not found</h1><p>{error || "This browser may have cleared its local TalonCV storage."}</p></section>;
  return <div className="shell results-page"><header className="results-heading"><div><div className="eyebrow">Local interview review</div><h1>{session.context.interviewQuestion}</h1><p>{session.context.targetRole || "General interview practice"} · stored only in this browser</p></div><label>Inference mode<select value={modelMode} disabled={running} onChange={(event) => setModelMode(event.target.value as typeof modelMode)}><option value="automatic">Automatic</option><option value="cpu">CPU / WASM</option><option value="webgpu">WebGPU when available</option></select></label></header>{recordingUrl && <div className="result-player"><video ref={playerRef} controls preload="metadata" src={recordingUrl} /><p>Replay stays local to this browser.</p></div>}{error && <p className="notice error">{error}</p>}{running ? <section className="progress-page"><span className="spinner" /><div className="eyebrow">{progress?.stage || "preparing"}</div><h1>{progress?.message || "Analyzing locally"}</h1><div className="large-progress"><span style={{ width: `${progress?.progress || 0}%` }} /></div><p>{progress?.progress || 0}% · You can keep this tab open while local workers process the recording.</p><button className="button ghost" onClick={() => controllerRef.current?.cancel()}>Cancel analysis</button></section> : !analysis ? <section className="centered-state"><h1>Ready for a local review</h1><p>Whisper, MiniLM, MediaPipe, deterministic scoring, and the optional learned cue model execute in this browser. Initial public model downloads are cached; recording content is never uploaded.</p><button className="button primary" onClick={() => void runAnalysis()}>Run local multimodal analysis</button></section> : <><div className="tabs" role="tablist" aria-label="Review sections">{reviewTabs.map((item) => <button key={item} role="tab" aria-selected={tab === item} onClick={() => setTab(item)}>{item}</button>)}</div><section className="tab-panel" role="tabpanel">{tab === "Overview" && <Overview analysis={analysis} seek={seek} />}{tab === "Transcript" && <div className="tab-stack"><header><h2>Transcript</h2><p>Generated by browser-local Whisper.</p></header><div className="transcript-list">{analysis.transcript.segments.map((segment, index) => <article key={`${segment.start}-${index}`}><button className="timestamp-button" onClick={() => seek(segment.start)}>{time(segment.start)}</button><p>{segment.text}</p><span>{time(segment.end)}</span></article>)}</div></div>}{tab === "Answer Quality" && <div className="tab-stack"><header><h2>Answer quality</h2><p>Deterministic response structure and optional local semantic relevance.</p></header><ScoreGrid analysis={{ ...analysis, scores: { scores: { verbalResponseQuality: asRecord(asRecord(analysis.scores).scores).verbalResponseQuality } } }} /><pre className="report-text">{JSON.stringify(analysis.responseAnalysis, null, 2)}</pre></div>}{tab === "Vocal Delivery" && <div className="tab-stack"><header><h2>Vocal delivery</h2><p>Waveform and timestamped transcript evidence.</p></header><EventList events={analysis.audioEvents} seek={seek} /><pre className="report-text">{JSON.stringify(analysis.audioFeatures, null, 2)}</pre></div>}{tab === "Visual Cues" && <div className="tab-stack"><header><h2>Visual cues</h2><p>Face detection, landmarks, posture, framing, and learned-cue additions are observable proxies, not personality or emotion claims.</p></header><EventList events={analysis.visualEvents} seek={seek} /><p className="fine-print">{analysis.visualFeatures.length} sampled frames analyzed locally.</p></div>}{tab === "Multimodal Moments" && <div className="tab-stack"><header><h2>Multimodal moments</h2><p>Timestamp alignment connects audio, transcript, and visual evidence.</p></header><Moments moments={analysis.moments} seek={seek} /></div>}{tab === "Full Report" && <div className="tab-stack"><header><h2>Full deterministic report</h2><p>The deterministic report stays complete even when optional local generation is unavailable.</p></header><pre className="report-text">{analysis.report}</pre><div className="coaching-block"><h2>Optional local coaching wording</h2>{analysis.localCoaching?.available ? <pre>{analysis.localCoaching.text}</pre> : <><p>{analysis.localCoaching?.warnings?.[0] || "Generate an additional summary with the optional browser-local model."}</p><button className="button secondary" onClick={() => void enhanceCoaching()}>Generate locally</button></>}</div></div>}{tab === "Downloads" && <div className="tab-stack"><header><h2>Downloads and local data</h2><p>Export a portable copy or remove this session from the browser history.</p></header><div className="download-list"><div><div><strong>Complete local review bundle</strong><span>Recording, report, transcript, analysis, and session context</span></div><button className="button secondary small" onClick={() => void downloadBundle()}>Download ZIP</button></div><div><div><strong>Run again</strong><span>Replaces the stored analysis with a new local result</span></div><button className="button ghost small" onClick={() => void runAnalysis()}>Reanalyze</button></div></div></div>}</section></>}</div>;
}
