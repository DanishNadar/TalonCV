export type RecorderState =
  | "idle"
  | "requesting_permissions"
  | "ready"
  | "recording"
  | "finalizing"
  | "uploading"
  | "uploaded"
  | "error";

export type RecorderEvent =
  | "REQUEST_PERMISSIONS"
  | "PERMISSIONS_GRANTED"
  | "START"
  | "STOP"
  | "FINALIZED"
  | "UPLOAD"
  | "UPLOADED"
  | "FAIL"
  | "RESET";

const transitions: Record<RecorderState, Partial<Record<RecorderEvent, RecorderState>>> = {
  idle: { REQUEST_PERMISSIONS: "requesting_permissions", FAIL: "error", RESET: "idle" },
  requesting_permissions: { PERMISSIONS_GRANTED: "ready", FAIL: "error", RESET: "idle" },
  ready: { START: "recording", UPLOAD: "uploading", FAIL: "error", RESET: "idle" },
  recording: { STOP: "finalizing", FAIL: "error" },
  finalizing: { FINALIZED: "ready", FAIL: "error" },
  uploading: { UPLOADED: "uploaded", FAIL: "error", RESET: "ready" },
  uploaded: { RESET: "ready" },
  error: { RESET: "idle", REQUEST_PERMISSIONS: "requesting_permissions" },
};

export function nextRecorderState(state: RecorderState, event: RecorderEvent): RecorderState {
  return transitions[state][event] ?? state;
}

export function preferredRecorderMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "video/webm";
  return ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"].find((type) =>
    MediaRecorder.isTypeSupported(type),
  ) ?? "video/webm";
}

export const maxRecordingSeconds = 300;
export const maxUploadBytes = 250 * 1024 * 1024;

export function permissionErrorMessage(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError") return "Camera or microphone permission was denied. Allow both devices in your browser's site settings and try again.";
  if (name === "NotFoundError") return "A camera or microphone was not found. Connect both devices or upload a recording instead.";
  return "TalonCV could not open the camera and microphone. Close other apps using them and try again.";
}
