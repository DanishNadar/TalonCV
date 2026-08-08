/// <reference lib="webworker" />
import { pipeline, type PretrainedModelOptions } from "@huggingface/transformers";
import { browserModels } from "@/config/browser-models";
import { configureLocalOnnxRuntime } from "@/lib/inference/onnxRuntime";
import type { WorkerRequest, WorkerResponse } from "@/lib/inference/workerProtocol";

declare const self: DedicatedWorkerGlobalScope;
configureLocalOnnxRuntime();
let cancelled = false;
function send(message: WorkerResponse) { self.postMessage(message); }
function prompt(evidence: Record<string, unknown>) { return `You are TalonCV's local interview-practice editor. Use only this structured evidence. Do not infer personality, emotion, anxiety, honesty, intelligence, employability, protected traits, or hiring suitability. Do not invent facts. Give a concise coaching summary with two strengths and up to three practice steps.\n\nEvidence:\n${JSON.stringify(evidence).slice(0, 9000)}\n\nCoaching summary:`; }

self.onmessage = async (message: MessageEvent<WorkerRequest>) => {
  const request = message.data;
  try {
    if (request.type === "cancel" || request.type === "dispose") { cancelled = true; send({ type: "ready", requestId: request.requestId }); return; }
    if (request.type !== "analyze") return; cancelled = false; const provider = request.mode === "webgpu" ? "webgpu" : "wasm";
    const model = browserModels.coach; send({ type: "progress", progress: { stage: "loadingCoach", progress: 0, message: "Loading optional local coaching model", modelId: model.id, totalBytes: model.estimatedBytes } });
    const options = { dtype: model.dtype, device: provider, revision: model.revision, progress_callback: (info: { progress?: number; loaded?: number; total?: number; file?: string }) => send({ type: "progress", progress: { stage: "downloadingModel", progress: Math.round(info.progress || 0), message: `Downloading ${info.file || model.label}`, modelId: model.id, loadedBytes: info.loaded, totalBytes: info.total } }) } as PretrainedModelOptions;
    let generator: Awaited<ReturnType<typeof pipeline>> | undefined;
    try { generator = await pipeline("text-generation", model.model!, options); if (cancelled) throw new Error("Coaching cancelled."); send({ type: "progress", progress: { stage: "generatingCoach", progress: 25, message: "Generating local coaching" } }); const output = await generator(prompt(request.payload as Record<string, unknown>), { max_new_tokens: 260, do_sample: false, return_full_text: false } as never) as unknown as Array<{ generated_text?: string }>; const text = output[0]?.generated_text?.trim(); if (!text) throw new Error("The local coaching model returned no text."); send({ type: "result", requestId: request.requestId, payload: { available: true, text, model: model.model } }); }
    finally { await (generator as { dispose?: () => Promise<void> } | undefined)?.dispose?.(); }
  } catch (error) { send({ type: "error", requestId: request.requestId, error: error instanceof Error ? error.message : "Local coaching could not be generated." }); }
};
