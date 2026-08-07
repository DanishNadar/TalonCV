import type { InferenceMode } from "@/types/local";

export type WorkerLifecycle = "load" | "progress" | "ready" | "analyze" | "cancel" | "dispose" | "error";
export interface WorkerProgress { stage: string; progress: number; message: string; modelId?: string; loadedBytes?: number; totalBytes?: number; }
export interface WorkerRequest { type: WorkerLifecycle; requestId?: string; payload?: unknown; mode?: InferenceMode; }
export interface WorkerResponse { type: WorkerLifecycle | "result"; requestId?: string; payload?: unknown; progress?: WorkerProgress; error?: string; }
