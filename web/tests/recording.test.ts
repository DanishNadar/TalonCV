import { describe, expect, it } from "vitest";
import { maxRecordingSeconds, maxUploadBytes, nextRecorderState, permissionErrorMessage } from "@/lib/recording";

describe("recording state machine", () => {
  it("covers the complete successful recording and upload flow", () => {
    let state = nextRecorderState("idle", "REQUEST_PERMISSIONS");
    expect(state).toBe("requesting_permissions");
    state = nextRecorderState(state, "PERMISSIONS_GRANTED");
    expect(state).toBe("ready");
    state = nextRecorderState(state, "START");
    expect(state).toBe("recording");
    state = nextRecorderState(state, "STOP");
    expect(state).toBe("finalizing");
    state = nextRecorderState(state, "FINALIZED");
    expect(state).toBe("ready");
    state = nextRecorderState(state, "UPLOAD");
    expect(state).toBe("uploading");
    expect(nextRecorderState(state, "UPLOADED")).toBe("uploaded");
  });

  it("surfaces actionable camera and microphone permission failures", () => {
    expect(permissionErrorMessage(new DOMException("denied", "NotAllowedError"))).toMatch(/permission was denied/i);
    expect(permissionErrorMessage(new DOMException("missing", "NotFoundError"))).toMatch(/not found/i);
    expect(nextRecorderState("requesting_permissions", "FAIL")).toBe("error");
  });

  it("enforces the public duration and upload limits", () => {
    expect(maxRecordingSeconds).toBe(300);
    expect(maxUploadBytes).toBe(250 * 1024 * 1024);
  });
});
