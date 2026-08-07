export type InferenceMode = "automatic" | "cpu" | "webgpu";
export type AnalysisState = "idle" | "preparing" | "downloading_model" | "analyzing" | "cancelled" | "complete" | "failed";

export interface LocalSessionContext {
  interviewQuestion: string;
  targetRole?: string;
  jobDescription?: string;
  desiredCompetencies?: string;
}

export interface LocalRecording {
  id: string;
  mimeType: string;
  durationSeconds: number;
  sizeBytes: number;
  hasAudio: boolean;
  hasVideo: boolean;
  createdAt: string;
}

export interface LocalSession {
  id: string;
  createdAt: string;
  updatedAt: string;
  context: LocalSessionContext;
  recording?: LocalRecording;
  analysisState: AnalysisState;
  analysisVersion?: string;
  modelVersions?: Record<string, string>;
  overallScore?: number;
  error?: string;
}

export interface TimedWord { text: string; start: number; end: number; confidence?: number | null; }
export interface TranscriptSegment { text: string; start: number; end: number; confidence?: number | null; words?: TimedWord[]; }
export interface TranscriptArtifact { text: string; segments: TranscriptSegment[]; averageConfidence: number | null; language?: string; available: boolean; warnings?: string[]; }
export interface EvidenceEvent {
  eventType: string;
  startTime: number;
  endTime: number;
  durationSeconds: number;
  explanation: string;
  coachingInterpretation?: string;
  description?: string;
  reliability?: "low" | "medium" | "high";
  confidence?: number;
  measurements?: Record<string, unknown>;
  provenance?: "rule" | "learned" | "combined" | "mediapipe";
}

export interface LocalAnalysis {
  analysisVersion: string;
  createdAt: string;
  mediaInfo: { durationSeconds: number; hasAudio: boolean; hasVideo: boolean; mimeType: string };
  sessionContext: LocalSessionContext;
  transcript: TranscriptArtifact;
  responseAnalysis: Record<string, unknown>;
  semanticAnalysis: Record<string, unknown>;
  audioFeatures: Record<string, unknown>;
  audioEvents: EvidenceEvent[];
  visualFeatures: Array<Record<string, unknown>>;
  visualEvents: EvidenceEvent[];
  moments: Array<Record<string, unknown>>;
  scores: Record<string, unknown>;
  report: string;
  localCoaching?: { available: boolean; text?: string; warnings?: string[]; model?: string };
  modelVersions: Record<string, string>;
  warnings: string[];
}

export interface StoredArtifact {
  sessionId: string;
  key: string;
  value: unknown;
  updatedAt: string;
}
