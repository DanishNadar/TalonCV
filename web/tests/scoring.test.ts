import { describe, expect, it } from "vitest";
import { buildCoachingScores } from "@/lib/inference/multimodal/scoring";
import { analyzeTranscript } from "@/lib/inference/audio/transcriptAnalyzer";
import { resampleForSpeech } from "@/lib/inference/audio/speechResampler";
import type { EvidenceEvent, TranscriptArtifact } from "@/types/local";

const transcriptOf = (text: string): TranscriptArtifact => ({
  text,
  segments: [{ text, start: 0, end: 40 }],
  averageConfidence: 0.9,
  available: true,
});

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

  it("withholds the overall score when speech was audible but never transcribed", () => {
    const result = buildCoachingScores(
      { hasAudio: true, hasVideo: true, durationSeconds: 90 },
      audio(),
      [],
      { available: false, metrics: { wordCount: 0 } },
      [event("cameraFacing", 0, 85), event("centeredFraming", 0, 85)],
      [],
    );
    const overall = overallOf(result);
    expect(overall.score).toBeNull();
    expect(overall.evidence.join(" ")).toMatch(/Transcription did not complete/);
  });

  it("withholds the overall score when a spoken response is too short to judge", () => {
    const result = buildCoachingScores(
      { hasAudio: true, hasVideo: true, durationSeconds: 30 },
      audio(),
      [],
      response({ metrics: { wordCount: 10, fillerRatePer100Words: 0 } }),
      [event("cameraFacing", 0, 30), event("centeredFraming", 0, 30)],
      [],
    );
    const overall = overallOf(result);
    expect(overall.score).toBeNull();
    expect(overall.evidence.join(" ")).toMatch(/Only 10 transcript words/);
  });

  it("withholds answer scoring when the transcript is repetitive and unreliable", () => {
    const result = buildCoachingScores(
      { hasAudio: true, hasVideo: true, durationSeconds: 30 },
      audio(),
      [],
      response({ reliableForContentScoring: false }),
      [event("cameraFacing", 0, 30)],
      [],
    );
    expect(dimensionOf(result, "verbalResponseQuality").score).toBeNull();
    expect(overallOf(result).score).toBeNull();
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

describe("answer content rubric", () => {
  const context = { interviewQuestion: "Tell me about a challenging technical project." };
  const onTopic =
    "For example, on a project last year I led a rebuild of our data pipeline. I implemented a batching strategy and as a result we reduced processing time by 37 percent and improved reliability for 12000 users.";
  const offTopic = "Um, yeah, so like, I guess, you know, basically I am a hard worker and a team player and a people person.";

  it("maps semantic similarity onto the score scale instead of ignoring it", () => {
    const relevant = analyzeTranscript(transcriptOf(onTopic), context, { available: true, questionRelevance: 0.44 });
    const irrelevant = analyzeTranscript(transcriptOf(onTopic), context, { available: true, questionRelevance: 0.05 });
    const relevantScore = (relevant.rubric as Record<string, { score: number }>).overallVerbalResponse.score;
    const irrelevantScore = (irrelevant.rubric as Record<string, { score: number }>).overallVerbalResponse.score;
    // Identical wording, different topical fit: the gap must be substantial.
    expect(relevantScore - irrelevantScore).toBeGreaterThan(25);
  });

  it("caps an answer that does not address the question", () => {
    const drifted = analyzeTranscript(transcriptOf(onTopic), context, { available: true, questionRelevance: 0.04 });
    expect((drifted.rubric as Record<string, { score: number }>).overallVerbalResponse.score).toBeLessThanOrEqual(38);
    expect((drifted.practiceAreas as string[]).join(" ")).toMatch(/Answer the question that was asked/);
  });

  it("rates vague filler-heavy content far below concrete content", () => {
    const good = analyzeTranscript(transcriptOf(onTopic), context, { available: true, questionRelevance: 0.4 });
    const bad = analyzeTranscript(transcriptOf(offTopic), context, { available: true, questionRelevance: 0.4 });
    const score = (result: Record<string, unknown>) => (result.rubric as Record<string, { score: number }>).overallVerbalResponse.score;
    expect(score(good) - score(bad)).toBeGreaterThan(20);
  });

  it("flags a transcript dominated by repeated short sounds as unreliable", () => {
    const repeated = analyzeTranscript(
      transcriptOf("I'm so sorry. Oh, oh, oh, oh, oh, oh."),
      context,
      { available: true, questionRelevance: 0.2 },
    );
    expect(repeated.reliableForContentScoring).toBe(false);
    expect((repeated.warnings as string[]).join(" ")).toMatch(/repeated short sounds/);
  });
});

describe("speech resampling", () => {
  it("converts browser-rate samples to Whisper's required 16 kHz", () => {
    const source = new Float32Array(48_000).fill(0.25);
    const output = resampleForSpeech(source, 48_000);
    expect(output).toHaveLength(16_000);
    expect(output[0]).toBeCloseTo(0.25, 6);
    expect(output[output.length - 1]).toBeCloseTo(0.25, 6);
  });

  it("does not alter audio that is already 16 kHz", () => {
    const source = new Float32Array([0.1, -0.2, 0.3]);
    expect(resampleForSpeech(source, 16_000)).toBe(source);
  });
});
