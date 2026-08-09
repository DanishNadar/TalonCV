import { describe, expect, it } from "vitest";
import { analyzeSemanticResponse } from "@/lib/inference/semantic/semanticAnalyzer";
import type { TranscriptArtifact } from "@/types/local";

const vector = (similarityToQuestion: number) => [similarityToQuestion, Math.sqrt(1 - similarityToQuestion ** 2)];

describe("semantic topic matching", () => {
  it("calibrates relevant introductions and evaluates substantive continuations with local context", async () => {
    const transcript: TranscriptArtifact = {
      available: true,
      text: "Hi, my name is Denjnadar. I am an artificial intelligence major at Illinois Tech. I work on several projects, mostly automation related to that. camera. Um, yep.",
      averageConfidence: 0.9,
      segments: [
        { start: 0, end: 5, text: "Hi, my name is Denjnadar. I am an artificial intelligence major at Illinois Tech." },
        { start: 5, end: 12, text: "I work on several projects, mostly automation related to that." },
        { start: 12, end: 16, text: "camera. Um, yep." },
      ],
    };
    const embeddingByText = new Map<string, number[]>([
      ["Tell me about yourself.", vector(1)],
      [transcript.segments[0].text, vector(0.28)],
      [transcript.segments[1].text, vector(0.09)],
      [transcript.segments[2].text, vector(0.14)],
      [`${transcript.segments[0].text} ${transcript.segments[1].text}`, vector(0.29)],
    ]);
    const result = await analyzeSemanticResponse(
      transcript,
      { interviewQuestion: "Tell me about yourself." },
      async (texts) => texts.map((text) => embeddingByText.get(text) || vector(0)),
    );
    const assessments = result.segmentAssessments as Array<Record<string, unknown>>;

    expect(assessments[0].topicMatchScore).toBeGreaterThanOrEqual(70);
    expect(assessments[1].topicMatchScore).toBeGreaterThanOrEqual(70);
    expect(assessments[1].relevanceBasis).toBe("local context + segment");
    expect(assessments[2].topicMatchScore).toBeLessThan(40);
  });

  it("does not confuse poor response quality with topic relevance", async () => {
    const transcript: TranscriptArtifact = {
      available: true,
      text: "I work on several projects, mostly automation related to that.",
      averageConfidence: 0.9,
      segments: [{ start: 0, end: 5, text: "I work on several projects, mostly automation related to that." }],
    };
    const result = await analyzeSemanticResponse(
      transcript,
      { interviewQuestion: "Tell me about yourself." },
      async (texts) => texts.map((text) => vector(text === "Tell me about yourself." ? 1 : 0.27)),
    );
    const assessment = (result.segmentAssessments as Array<Record<string, unknown>>)[0];

    expect(assessment.topicMatchScore).toBeGreaterThanOrEqual(70);
  });
});
