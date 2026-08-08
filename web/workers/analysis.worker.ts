/// <reference lib="webworker" />
import { pipeline, type PretrainedModelOptions } from "@huggingface/transformers";
import { browserModels } from "@/config/browser-models";
import { analyzeAudio } from "@/lib/inference/audio/audioAnalyzer";
import { analyzeTranscript } from "@/lib/inference/audio/transcriptAnalyzer";
import { analyzeSemanticResponse } from "@/lib/inference/semantic/semanticAnalyzer";
import { configureLocalOnnxRuntime } from "@/lib/inference/onnxRuntime";
import type { LocalSessionContext, TranscriptArtifact } from "@/types/local";
import type { WorkerRequest, WorkerResponse } from "@/lib/inference/workerProtocol";

declare const self: DedicatedWorkerGlobalScope;
configureLocalOnnxRuntime();
let cancelled = false;
type LocalPipeline = ((input: unknown, options?: unknown) => Promise<unknown>) & { dispose?: () => Promise<void> };
function send(message: WorkerResponse) { self.postMessage(message); }
function progress(stage: string, percent: number, message: string, modelId?: string, loadedBytes?: number, totalBytes?: number) { send({ type: "progress", progress: { stage, progress: percent, message, modelId, loadedBytes, totalBytes } }); }
// ONNX Runtime Web's extended graph optimizer runs a QDQ transform
// (TransposeDQWeightsForMatMulNBits) that rejects the merged Whisper decoder and
// aborts session creation for every dtype. Capping optimization at "basic"
// skips that pass; the model still runs, it is just not re-fused.
const sessionOptions = { graphOptimizationLevel: "basic" };

function modelOptions(model: typeof browserModels.speechFast, provider: "wasm" | "webgpu"): PretrainedModelOptions {
  return { dtype: model.dtype, device: provider, revision: model.revision, session_options: sessionOptions, progress_callback: (info: { status?: string; file?: string; loaded?: number; total?: number; progress?: number }) => progress("downloadingModel", Math.round(info.progress ?? 0), info.status === "progress" ? `Downloading ${info.file || model.label}` : `Loading ${model.label}`, model.id, info.loaded, info.total) } as PretrainedModelOptions;
}
function toSegments(output: unknown): TranscriptArtifact {
  const result = output as { text?: string; chunks?: Array<{ text?: string; timestamp?: [number | null, number | null] }> }; const chunks = result.chunks || []; const segments = chunks.map((chunk, index) => ({ text: chunk.text?.trim() || "", start: chunk.timestamp?.[0] ?? index * 2, end: chunk.timestamp?.[1] ?? (index + 1) * 2 })).filter((segment) => segment.text);
  return { available: Boolean(result.text?.trim()), text: result.text?.trim() || segments.map((segment) => segment.text).join(" "), segments, averageConfidence: null, language: "en", warnings: segments.length ? [] : ["Whisper did not return timestamped segments."] };
}
/** WebGPU can be advertised and still fail to produce a backend at load time or
 *  a device at run time. CPU/WASM always works, so any accelerated attempt falls
 *  back to it rather than losing the whole modality. */
async function withCpuFallback<T>(provider: "wasm" | "webgpu", run: (provider: "wasm" | "webgpu") => Promise<T>): Promise<T> {
  try {
    return await run(provider);
  } catch (error) {
    if (provider !== "webgpu" || cancelled) throw error;
    progress("retryingOnCpu", 0, "Accelerated backend unavailable, retrying on CPU");
    return run("wasm");
  }
}

/** Quantized weight variants are not uniformly loadable: a given ONNX Runtime
 *  Web build can reject one dtype's decoder graph outright while another loads
 *  fine. Transcription therefore walks a chain of (dtype, device) pairs and
 *  keeps the first that produces a session, instead of losing speech entirely. */
async function transcribe(samples: Float32Array, provider: "wasm" | "webgpu", profile: "fast" | "balanced"): Promise<TranscriptArtifact> {
  const preferred = profile === "balanced" ? browserModels.speechBalanced : browserModels.speechFast;
  const dtypes = [preferred.dtype!, preferred.dtype === "q8" ? "q4" : "q8", "fp32"];
  // Speech runs on CPU/WASM only. WebGPU has failed to produce a usable backend
  // in every environment tested, and for a model this small the accelerator buys
  // little while its failure mode poisons ORT's backend registry for the worker.
  void provider;
  const attempts: Array<{ dtype: string; device: "wasm" | "webgpu" }> = dtypes.map((dtype) => ({ dtype, device: "wasm" as const }));

  const failures: string[] = [];
  for (const [index, attempt] of attempts.entries()) {
    if (cancelled) throw new Error("Analysis cancelled.");
    let pipe: LocalPipeline | undefined;
    try {
      progress("loadingSpeech", 0, "Loading speech model locally", preferred.id, 0, preferred.estimatedBytes);
      pipe = await pipeline("automatic-speech-recognition", preferred.model!, modelOptions({ ...preferred, dtype: attempt.dtype }, attempt.device)) as unknown as LocalPipeline;
      if (cancelled) throw new Error("Analysis cancelled.");
      progress("transcribing", 5, "Transcribing locally", preferred.id, preferred.estimatedBytes, preferred.estimatedBytes);
      // whisper-tiny.en is English-only, so `language`/`task` are rejected, and
      // the export carries no cross-attentions, so word-level timestamps are
      // unavailable. Segment chunks are what downstream analysis expects anyway.
      const output = await pipe(samples, { return_timestamps: true, chunk_length_s: 30, stride_length_s: 5 });
      return toSegments(output);
    } catch (error) {
      if (cancelled) throw error;
      failures.push(`${attempt.dtype}/${attempt.device}: ${error instanceof Error ? error.message : String(error)}`);
      if (index < attempts.length - 1) progress("retryingSpeech", 0, "Trying another local speech configuration");
    } finally {
      await pipe?.dispose?.();
    }
  }
  throw new Error(`Local speech transcription failed. Attempts — ${failures.join(" | ")}`);
}
async function runSemantic(transcript: TranscriptArtifact, context: LocalSessionContext, provider: "wasm" | "webgpu") {
  const model = browserModels.semantic;
  void provider;
  return withCpuFallback("wasm", async (device) => {
    progress("loadingSemantic", 0, "Loading semantic model locally", model.id, 0, model.estimatedBytes);
    let pipe: LocalPipeline | undefined;
    try { pipe = await pipeline("feature-extraction", model.model!, modelOptions(model, device)) as unknown as LocalPipeline; if (cancelled) throw new Error("Analysis cancelled."); progress("analyzingSemantic", 10, "Analyzing relevance locally", model.id, model.estimatedBytes, model.estimatedBytes); return await analyzeSemanticResponse(transcript, context, async (texts) => { const output = await pipe!(texts, { pooling: "mean", normalize: true }) as { tolist?: () => number[][]; data?: Float32Array; dims?: number[] }; if (output.tolist) return output.tolist(); const dimensions = output.dims?.at(-1) || 1; return Array.from(output.data || []).reduce<number[][]>((all, value, index) => { const bucket = Math.floor(index / dimensions); (all[bucket] ??= []).push(value); return all; }, []); }); }
    finally { await pipe?.dispose?.(); }
  });
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
