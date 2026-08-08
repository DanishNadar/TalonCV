export type BrowserModelId = "speechFast" | "speechBalanced" | "semantic" | "vision" | "coach";

export interface BrowserModelDefinition {
  id: BrowserModelId;
  label: string;
  model?: string;
  revision?: string;
  dtype?: string;
  estimatedBytes: number;
  optional?: boolean;
  runtime: "transformers" | "mediapipe" | "onnx";
  files?: Array<{ url: string; bytes: number }>;
}

// Revisions and byte sizes were verified from the public static model manifests
// on 2026-08-07. They are provenance metadata, never credentials or API routes.
export const browserModels: Record<BrowserModelId, BrowserModelDefinition> = {
  speechFast: {
    id: "speechFast", label: "Speech model (Fast)", model: "onnx-community/whisper-tiny.en",
    revision: "2575352d61be1bf7225cf8f8b268a4678025fc58", dtype: "q4", estimatedBytes: 99_250_000, runtime: "transformers",
  },
  speechBalanced: {
    id: "speechBalanced", label: "Speech model (Balanced)", model: "onnx-community/whisper-tiny.en",
    revision: "2575352d61be1bf7225cf8f8b268a4678025fc58", dtype: "q8", estimatedBytes: 43_450_000, runtime: "transformers",
  },
  semantic: {
    id: "semantic", label: "Semantic model", model: "Xenova/all-MiniLM-L6-v2",
    revision: "751bff37182d3f1213fa05d7196b954e230abad9", dtype: "q8", estimatedBytes: 24_000_000, runtime: "transformers",
  },
  vision: {
    id: "vision", label: "Visual models", estimatedBytes: 9_766_088, runtime: "mediapipe",
    files: [
      { url: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", bytes: 229_746 },
      { url: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", bytes: 3_758_596 },
      { url: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task", bytes: 5_777_746 },
    ],
  },
  coach: {
    id: "coach", label: "Local coaching model", model: "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA",
    revision: "5b6682c7c9df18f004bfb7e635cba3f3d98537d8", dtype: "q4f16", estimatedBytes: 120_000_000, optional: true, runtime: "transformers",
  },
};

export const browserAnalysisVersion = "browser-local-v2";
export const modelHostPolicy = {
  purpose: "Static model files only; TalonCV never sends interview content to a model host.",
  localFaceOnnxUrl: "/models/yolo11n-face.onnx",
  localCueClassifierUrl: "/models/cue-classifier.json",
};

export function formatBytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(bytes >= 100_000_000 ? 0 : 1)} MB`;
}
