import { describe, expect, it, vi } from "vitest";
import { reviewTabs, seekVideoElement } from "@/components/LocalAnalysisWorkspace";

describe("results navigation", () => {
  it("preserves all eight TalonCV review tabs", () => {
    expect(reviewTabs).toEqual([
      "Overview", "Transcript", "Answer Quality", "Vocal Delivery",
      "Visual Cues", "Multimodal Moments", "Full Report", "Export",
    ]);
  });

  it("seeks the private replay player from a timestamp", () => {
    const video = document.createElement("video");
    Object.defineProperty(video, "play", { value: vi.fn().mockResolvedValue(undefined) });
    expect(seekVideoElement(video, 84.25)).toBe(true);
    expect(video.currentTime).toBe(84.25);
    expect(video.play).toHaveBeenCalled();
  });
});
