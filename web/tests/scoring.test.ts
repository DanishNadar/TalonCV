import { describe, expect, it } from "vitest";
import { buildCoachingScores } from "@/lib/inference/multimodal/scoring";
import type { EvidenceEvent } from "@/types/local";

const event = (eventType: string, startTime: number, endTime: number): EvidenceEvent => ({
  eventType,
  startTime,
  endTime,
  durationSeconds: endTime - startTime,
  explanation: "",
});

const audio = (over: Record<string, unknown> = {}) => ({
  available: true,
  durationSeconds: 90,
  speechRatio: 0.72,
  silenceRatio: 0.28,
  overallRmsDb: -20,
  snrProxyDb: 26,
  clippingPercentage: 0.001,
  dropoutRatio: 0,
  speechRateWpm: 140,
  energyVariationDb: 12,
  volumeConsistencyStdDb: 3,
  ...over,
});

const response = (over: Record<string, unknown> = {}) => ({
  available: true,
  confidence: "medium",
  metrics: { wordCount: 180, fillerRatePer100Words: 1 },
  starAnalysis: { componentsPresent: 4 },
  rubric: { overallVerbalResponse: { score: 88 }, specificity: { score: 85 }, clarity: { score: 90 } },
  practiceAreas: [],
  ...over,
});

const overallOf = (result: Record<string, unknown>) =>
  (result.scores as Record<string, { score: number | null; rating: string; evidence: string[] }>).overallInterviewPracticeDelivery;
const dimensionOf = (result: Record<string, unknown>, key: string) =>
  (result.scores as Record<string, { score: number | null; rating: string }>)[key];

describe("browser coaching scores", () => {
  it("withholds an overall score for a clip too short to judge", () => {
    const result = buildCoachingScores(
      { hasAudio: true, hasVideo: true, durationSeconds: 1 },
      audio({ durationSeconds: 1, speechRatio: 0.05 }),
      [],
      { available: false, metrics: { wordCount: 0 } },
      [event("cameraFacing", 0, 1)],
      [],
    );
    const overall = overallOf(result);
    expect(overall.score).toBeNull();
    expect(overall.rating).toBe("Insufficient evidence");
    expect(overall.evidence.join(" ")).toMatch(/at least 10s/);
  });

  it("scores a silent take as a failure to answer rather than excluding it", () => {
    const result = buildCoachingScores(
      { hasAudio: true, hasVideo: true, durationSeconds: 60 },
      audio({ speechRatio: 0.02 }),
      [],
      { available: false, metrics: { wordCount: 0 } },
      [event("cameraFacing", 0, 60)],
      [],
    );
    expect(dimensionOf(result, "verbalResponseQuality").score).toBeLessThan(20);
    expect(dimensionOf(result, "vocalDelivery").score).toBeLessThan(20);
    expect(overallOf(result).score).toBeLessThan(45);
  });

  it("rewards a strong take with a high score", () => {
    const result = buildCoachingScores(
      { hasAudio: true, hasVideo: true, durationSeconds: 90 },
      audio(),
      [],
      response(),
      [event("cameraFacing", 0, 80), event("centeredFraming", 0, 85), event("stablePosture", 0, 80)],
      [{ classification: "strength", explanation: "", coachingRecommendation: "" }],
    );
    const overall = overallOf(result);
    expect(overall.score).not.toBeNull();
    expect(overall.score!).toBeGreaterThan(82);
  });

  it("separates a weak take from a strong one by a wide margin", () => {
    const strong = overallOf(
      buildCoachingScores(
        { hasAudio: true, hasVideo: true, durationSeconds: 90 },
        audio(),
        [],
        response(),
        [event("cameraFacing", 0, 85), event("centeredFraming", 0, 85)],
        [{ classification: "strength", explanation: "", coachingRecommendation: "" }],
      ),
    ).score!;

    const weak = overallOf(
      buildCoachingScores(
        { hasAudio: true, hasVideo: true, durationSeconds: 90 },
        audio({ speechRateWpm: 230, energyVariationDb: 2, snrProxyDb: 5, clippingPercentage: 2, overallRmsDb: -46 }),
        Array.from({ length: 9 }, (_, index) => event("longPause", index * 9, index * 9 + 3)),
        response({ metrics: { wordCount: 60, fillerRatePer100Words: 12 }, rubric: { overallVerbalResponse: { score: 38 } }, practiceAreas: ["Add a concrete example."] }),
        [event("lookingAway", 0, 40), event("faceMissing", 40, 70), event("postureShift", 10, 50)],
        [{ classification: "review", explanation: "", coachingRecommendation: "Slow down." }],
      ),
    ).score!;

    expect(weak).toBeLessThan(55);
    expect(strong - weak).toBeGreaterThan(25);
  });

  it("keeps visual scoring proportional to time rather than event count", () => {
    const brief = dimensionOf(
      buildCoachingScores(
        { hasAudio: true, hasVideo: true, durationSeconds: 120 },
        audio(),
        [],
        response(),
        [event("lookingAway", 0, 4)],
        [],
      ),
      "visualDelivery",
    ).score!;
    const sustained = dimensionOf(
      buildCoachingScores(
        { hasAudio: true, hasVideo: true, durationSeconds: 120 },
        audio(),
        [],
        response(),
        [event("lookingAway", 0, 90)],
        [],
      ),
      "visualDelivery",
    ).score!;
    expect(brief).toBeGreaterThan(sustained + 20);
  });
});
