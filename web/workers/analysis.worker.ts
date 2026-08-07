/// <reference lib="webworker" />
import { pipeline, type PretrainedModelOptions } from "@huggingface/transformers";
import { browserModels } from "@/config/browser-models";
import { analyzeAudio } from "@/lib/inference/audio/audioAnalyzer";
import { analyzeTranscript } from "@/lib/inference/audio/transcriptAnalyzer";
import { analyzeSemanticResponse } from "@/lib/inference/semantic/semanticAnalyzer";
import type { LocalSessionContext, TranscriptArtifact } from "@/types/local";
import type { WorkerRequest, WorkerResponse } from "@/lib/inference/workerProtocol";

declare const self: DedicatedWorkerGlobalScope;
let cancelled = false;
type LocalPipeline = ((input: unknown, options?: unknown) => Promise<unknown>) & { dispose?: () => Promise<void> };
function send(message: WorkerResponse) { self.postMessage(message); }
function progress(stage: string, percent: number, message: string, modelId?: string, loadedBytes?: number, totalBytes?: number) { send({ type: "progress", progress: { stage, progress: percent, message, modelId, loadedBytes, totalBytes } }); }
function modelOptions(model: typeof browserModels.speechFast, provider: "wasm" | "webgpu"): PretrainedModelOptions {
  return { dtype: model.dtype, device: provider, revision: model.revision, progress_callback: (info: { status?: string; file?: string; loaded?: number; total?: number; progress?: number }) => progress("downloadingModel", Math.round(info.progress ?? 0), info.status === "progress" ? `Downloading ${info.file || model.label}` : `Loading ${model.label}`, model.id, info.loaded, info.total) } as PretrainedModelOptions;
}
function toSegments(output: unknown): TranscriptArtifact {
  const result = output as { text?: string; chunks?: Array<{ text?: string; timestamp?: [number | null, number | null] }> }; const chunks = result.chunks || []; const segments = chunks.map((chunk, index) => ({ text: chunk.text?.trim() || "", start: chunk.timestamp?.[0] ?? index * 2, end: chunk.timestamp?.[1] ?? (index + 1) * 2 })).filter((segment) => segment.text);
  return { available: Boolean(result.text?.trim()), text: result.text?.trim() || segments.map((segment) => segment.text).join(" "), segments, averageConfidence: null, language: "en", warnings: segments.length ? [] : ["Whisper did not return timestamped segments."] };
}
async function transcribe(samples: Float32Array, provider: "wasm" | "webgpu", profile: "fast" | "balanced"): Promise<TranscriptArtifact> {
  const model = profile === "balanced" ? browserModels.speechBalanced : browserModels.speechFast; progress("loadingSpeech", 0, "Loading speech model locally", model.id, 0, model.estimatedBytes);
  let pipe: LocalPipeline | undefined;
  try { pipe = await pipeline("automatic-speech-recognition", model.model!, modelOptions(model, provider)) as unknown as LocalPipeline; if (cancelled) throw new Error("Analysis cancelled."); progress("transcribing", 5, "Transcribing locally", model.id, model.estimatedBytes, model.estimatedBytes); const output = await pipe(samples, { return_timestamps: "word", chunk_length_s: 30, stride_length_s: 5, language: "english", task: "transcribe" }); return toSegments(output); }
  finally { await pipe?.dispose?.(); }
}
async function runSemantic(transcript: TranscriptArtifact, context: LocalSessionContext, provider: "wasm" | "webgpu") {
  const model = browserModels.semantic; progress("loadingSemantic", 0, "Loading semantic model locally", model.id, 0, model.estimatedBytes);
  let pipe: LocalPipeline | undefined;
  try { pipe = await pipeline("feature-extraction", model.model!, modelOptions(model, provider)) as unknown as LocalPipeline; if (cancelled) throw new Error("Analysis cancelled."); progress("analyzingSemantic", 10, "Analyzing relevance locally", model.id, model.estimatedBytes, model.estimatedBytes); return await analyzeSemanticResponse(transcript, context, async (texts) => { const output = await pipe!(texts, { pooling: "mean", normalize: true }) as { tolist?: () => number[][]; data?: Float32Array; dims?: number[] }; if (output.tolist) return output.tolist(); const dimensions = output.dims?.at(-1) || 1; return Array.from(output.data || []).reduce<number[][]>((all, value, index) => { const bucket = Math.floor(index / dimensions); (all[bucket] ??= []).push(value); return all; }, []); }); }
  finally { await pipe?.dispose?.(); }
}

self.onmessage = async (message: MessageEvent<WorkerRequest>) => {
  const request = message.data;
  try {
    if (request.type === "cancel") { cancelled = true; return; }
    if (request.type === "dispose") { cancelled = true; send({ type: "ready", requestId: request.requestId }); return; }
    if (request.type !== "analyze") return;
    cancelled = false;
    const payload = request.payload as { samples: ArrayBuffer; sampleRate: number; channels: number; context: LocalSessionContext; provider: "wasm" | "webgpu"; profile: "fast" | "balanced"; testTranscript?: TranscriptArtifact };
    const samples = new Float32Array(payload.samples); let transcript: TranscriptArtifact;
    try { transcript = payload.testTranscript || await transcribe(samples, payload.provider, payload.profile); }
    catch (error) { if (cancelled) throw error; transcript = { available: false, text: "", segments: [], averageConfidence: null, warnings: [error instanceof Error ? error.message : "Local speech transcription failed."] }; }
    if (cancelled) throw new Error("Analysis cancelled."); progress("analyzingAudio", 35, "Analyzing vocal delivery", undefined); const audio = analyzeAudio(samples, payload.sampleRate, payload.channels, transcript);
    let semanticAnalysis: Record<string, unknown> = { available: false, warnings: ["Semantic analysis unavailable."] };
    if (transcript.text) try { semanticAnalysis = await runSemantic(transcript, payload.context, payload.provider); } catch (error) { semanticAnalysis = { available: false, warnings: [error instanceof Error ? error.message : "Local semantic analysis failed."] }; }
    if (cancelled) throw new Error("Analysis cancelled."); progress("analyzingResponse", 82, "Analyzing answer structure", undefined); const response = analyzeTranscript(transcript, payload.context, semanticAnalysis); send({ type: "result", requestId: request.requestId, payload: { transcript, audioFeatures: audio.features, audioEvents: audio.events, semanticAnalysis, responseAnalysis: response } });
  } catch (error) { send({ type: "error", requestId: request.requestId, error: error instanceof Error ? error.message : "Local analysis failed." }); }
};
