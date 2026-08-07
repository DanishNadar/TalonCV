import fixture from "./fixtures/cue-classifier-parity.json";
import { describe, expect, it } from "vitest";
import { predictBrowserCue, type BrowserCueClassifier } from "@/lib/inference/vision/cueClassifier";
import type { VisualFeatureRow } from "@/lib/inference/vision/cueRules";

describe("browser random-forest cue classifier", () => {
  it("matches the sklearn predictions exported for the same fixture rows", () => {
    const classifier = fixture.classifier as BrowserCueClassifier;
    fixture.rows.forEach((row, index) => {
      const actual = predictBrowserCue({ timestampSeconds: index, ...row } as unknown as VisualFeatureRow, classifier);
      const expected = fixture.expected[index];
      expect(actual.candidate).toBe(expected.candidate);
      expect(actual.confidence).toBeCloseTo(expected.confidence, 10);
      Object.entries(expected.probabilities).forEach(([label, probability]) => expect(actual.probabilities[label]).toBeCloseTo(probability, 10));
    });
  });
});
