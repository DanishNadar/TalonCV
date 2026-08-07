export type JobStatus = "queued" | "claimed" | "processing" | "complete" | "failed" | "cancelled";

export interface AnalysisJob {
  id: string;
  recording_id: string;
  session_id: string;
  user_id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  error_code?: string | null;
  error_message?: string | null;
  cancellation_requested_at?: string | null;
  analysis_version: string;
  created_at: string;
}

export interface AnalysisArtifact {
  id: string;
  artifact_type: string;
  storage_path: string;
  content_type: string;
  size_bytes: number;
}

export interface AnalysisPayload {
  analysisVersion?: string;
  mediaInfo?: Record<string, unknown>;
  sessionContext?: Record<string, unknown>;
  transcript?: Record<string, unknown> & { text?: string; segments?: Array<Record<string, unknown>> };
  responseAnalysis?: Record<string, unknown>;
  semanticAnalysis?: Record<string, unknown>;
  audioFeatures?: Record<string, unknown>;
  audioEvents?: Array<Record<string, unknown>>;
  visualEvents?: Array<Record<string, unknown>>;
  moments?: Array<Record<string, unknown>>;
  scores?: Record<string, unknown>;
  localCoaching?: Record<string, unknown>;
  enhancedCoachingStatus?: string;
  warnings?: string[];
}
