import type { InferenceMode } from "@/types/local";
import type { WorkerRequest, WorkerResponse } from "@/lib/inference/workerProtocol";

export async function generateLocalCoaching(evidence: Record<string, unknown>, mode: InferenceMode, onProgress?: (message: string) => void): Promise<{ available: boolean; text?: string; model?: string; warnings?: string[] }> {
  const worker = new Worker(new URL("../../../workers/coaching.worker.ts", import.meta.url)); const requestId = crypto.randomUUID();
  try { return await new Promise((resolve, reject) => { worker.onmessage = (event: MessageEvent<WorkerResponse>) => { const response = event.data; if (response.type === "progress") onProgress?.(response.progress?.message || "Generating local coaching"); if (response.type === "result") resolve(response.payload as { available: boolean; text?: string; model?: string }); if (response.type === "error") reject(new Error(response.error)); }; worker.postMessage({ type: "analyze", requestId, payload: evidence, mode } satisfies WorkerRequest); }); }
  catch (error) { return { available: false, warnings: [error instanceof Error ? error.message : "Optional local coaching could not be generated."] }; }
  finally { worker.terminate(); }
}
